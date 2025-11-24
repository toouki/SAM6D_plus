from ultralytics import YOLO
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import logging
from .fast_sam import CustomYOLO

class Yolov11Seg:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: dict = None,
        segmentor_width_size=None,
        device=None,
    ):
        # 初始化配置参数，设置默认值防止None
        self.config = config or {}
        self.iou_threshold = self.config.get('iou_threshold', 0.5)
        self.conf_threshold = self.config.get('conf_threshold', 0.25)
        self.max_det = self.config.get('max_det', 200)
        
        # 新增：从配置读取目标类别，默认第一类（索引0）
        self.selected_classes = self._validate_selected_classes(
            self.config.get('selected_classes', [0])
        )
        logging.info(f"Selected classes for segmentation: {self.selected_classes}")
        
        # 验证并设置segmentor_width_size，防止无效值
        self.segmentor_width_size = self._validate_segmentor_size(segmentor_width_size)
        
        # 初始化设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = CustomYOLO(
            model=str(checkpoint_path),
            iou=self.iou_threshold,
            conf=self.conf_threshold,
            max_det=self.max_det,
            selected_device=self.device,
            segmentor_width_size=self.segmentor_width_size,
        )
        
        logging.info(f"Initialized YOLOv11Seg with model {checkpoint_path} on {self.device}")

    def _validate_selected_classes(self, classes: Any) -> List[int]:
        """验证选定的分割类别是否有效，确保为非负整数列表"""
        if not isinstance(classes, list):
            logging.warning(f"selected_classes must be a list, got {type(classes)}, using default [0]")
            return [0]
        
        valid_classes = []
        for c in classes:
            if isinstance(c, int) and c >= 0:
                valid_classes.append(c)
            else:
                logging.warning(f"Invalid class {c} (must be non-negative integer), skipping")
        
        if not valid_classes:
            logging.warning("No valid classes in selected_classes, using default [0]")
            return [0]
        
        return valid_classes

    def _validate_segmentor_size(self, size: Union[int, None]) -> int:
        """验证segmentor宽度尺寸，防止无效值导致后续计算错误"""
        if size is None:
            return 640  # 默认值
        if not isinstance(size, int) or size <= 0 or size > 4096:
            logging.warning(f"Invalid segmentor_width_size {size}, using default 640")
            return 640
        return size

    def _clamp_boxes(self, boxes: torch.Tensor, orig_size: Tuple[int, int]) -> torch.Tensor:
        """将边界框坐标限制在图像有效范围内"""
        h, w = orig_size
        # 确保坐标在有效范围内 (0 <= x <= w-1, 0 <= y <= h-1)
        boxes[:, 0] = torch.clamp(boxes[:, 0], 0, w - 1)  # x1
        boxes[:, 1] = torch.clamp(boxes[:, 1], 0, h - 1)  # y1
        boxes[:, 2] = torch.clamp(boxes[:, 2], 0, w - 1)  # x2
        boxes[:, 3] = torch.clamp(boxes[:, 3], 0, h - 1)  # y2
        
        # 确保x2 > x1且y2 > y1，防止无效边界框
        boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1)
        
        return boxes

    def postprocess_resize(self, detections, orig_size: Tuple[int, int], update_boxes: bool = False) -> Dict[str, Any]:
        """后处理调整掩码和边界框尺寸，添加严格的范围检查"""
        if detections["masks"].numel() == 0:  # 空掩码处理
            return detections
            
        # 验证原始尺寸有效性
        if not (isinstance(orig_size, tuple) and len(orig_size) == 2 and 
                all(isinstance(d, int) and d > 0 for d in orig_size)):
            raise ValueError(f"Invalid original size: {orig_size}, must be (height, width) with positive integers")
        
        # 调整掩码尺寸
        try:
            detections["masks"] = F.interpolate(
                detections["masks"].unsqueeze(1).float(),
                size=(orig_size[0], orig_size[1]),
                mode="bilinear",
                align_corners=False,
            )[:, 0, :, :]
        except Exception as e:
            logging.error(f"Error resizing masks: {e}")
            raise
        
        # 调整边界框并限制范围
        if update_boxes and "boxes" in detections and detections["boxes"].numel() > 0:
            # 计算缩放比例并检查有效性
            scale = orig_size[1] / self.segmentor_width_size
            if scale <= 0 or scale > 100:  # 限制极端缩放比例
                raise ValueError(f"Invalid scale factor {scale}, check segmentor width and original size")
            
            # 缩放边界框并限制范围
            detections["boxes"] = detections["boxes"].float() * scale
            detections["boxes"] = self._clamp_boxes(detections["boxes"], orig_size)
            
            # 转换为int32避免int64溢出
            detections["boxes"] = detections["boxes"].to(dtype=torch.int32)
        
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> Dict[str, Any]:
        """生成掩码和边界框，添加完整的错误处理和范围检查"""
        # 验证输入图像
        if not isinstance(image, np.ndarray) or len(image.shape) not in (2, 3):
            raise ValueError(f"Invalid image format: {image.shape}, expected (H, W) or (H, W, C)")
        
        orig_size = image.shape[:2]  # (H, W)
        resized_image = image
        
        # 调整图像尺寸（如果需要）
        if self.segmentor_width_size is not None:
            try:
                # 计算合理的缩放尺寸，避免极端值
                ratio = self.segmentor_width_size / orig_size[1]
                new_height = int(orig_size[0] * ratio)
                new_height = max(32, min(new_height, 8192))  # 限制高度范围
                
                resized_image = cv2.resize(image, (self.segmentor_width_size, new_height))
            except Exception as e:
                logging.error(f"Error resizing image: {e}")
                raise
        
        # 模型推理
        try:
            detections = self.model(resized_image)
            if not detections or len(detections) == 0:
                return {"masks": torch.empty((0,), device=self.device), "boxes": torch.empty((0, 4), device=self.device)}
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            raise
        
        # 提取掩码和边界框，并按选定类别过滤
        try:
            # 确保掩码和边界框存在且格式正确
            if not hasattr(detections[0], 'masks') or detections[0].masks is None:
                return {"masks": torch.empty((0,), device=self.device), "boxes": torch.empty((0, 4), device=self.device)}
                
            # 获取原始检测结果
            boxes_obj = detections[0].boxes
            classes = boxes_obj.cls  # 类别索引 [N]
            masks = detections[0].masks.data  # [N, H, W]
            boxes_xyxy = boxes_obj.data[:, :4]  # [N, 4] - xyxy格式
            
            # 新增：按选定类别过滤
            selected_classes_tensor = torch.tensor(
                self.selected_classes, 
                device=classes.device, 
                dtype=classes.dtype
            )
            # 找到属于选定类别的索引
            indices = torch.isin(classes, selected_classes_tensor)
            
            # 应用过滤
            masks = masks[indices]
            boxes_xyxy = boxes_xyxy[indices]
            
            # 过滤空结果
            if masks.numel() == 0 or boxes_xyxy.numel() == 0:
                return {"masks": torch.empty((0,), device=self.device), "boxes": torch.empty((0, 4), device=self.device)}
                
            # 转换为合适的数据类型，避免溢出
            masks = masks.to(device=self.device)
            boxes = boxes_xyxy.to(device=self.device, dtype=torch.float32)
            
        except Exception as e:
            logging.error(f"Error processing detection results: {e}")
            raise
        
        mask_data = {
            "masks": masks,
            "boxes": boxes,
        }
        
        # 后处理调整尺寸
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size, update_boxes=True)
        
        # 最终检查确保没有异常值
        self._validate_outputs(mask_data, orig_size)
        
        return mask_data

    def _validate_outputs(self, outputs: Dict[str, torch.Tensor], orig_size: Tuple[int, int]) -> None:
        """验证输出数据的有效性，防止后续处理出错"""
        h, w = orig_size
        
        # 检查掩码尺寸
        if "masks" in outputs and outputs["masks"].numel() > 0:
            mask_shape = outputs["masks"].shape[1:]  # [H, W]
            if mask_shape != (h, w):
                logging.warning(f"Mask shape {mask_shape} does not match original size {(h, w)}")
        
        # 检查边界框范围
        if "boxes" in outputs and outputs["boxes"].numel() > 0:
            boxes = outputs["boxes"]
            # 检查是否有超出范围的坐标（允许微小误差）
            if torch.any(boxes < -1) or torch.any(boxes[:, 0] > w) or torch.any(boxes[:, 2] > w) or \
               torch.any(boxes[:, 1] > h) or torch.any(boxes[:, 3] > h):
                logging.warning(f"Detected boxes outside image bounds: {boxes}")
                # 再次裁剪边界框
                outputs["boxes"] = self._clamp_boxes(boxes, orig_size)