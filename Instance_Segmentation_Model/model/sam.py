from ultralytics import SAM as UltralyticsSAM  # 避免命名冲突
import numpy as np
import torch
import torch.nn.functional as F
import logging
import os.path as osp
import cv2
from typing import Any, Dict, List


class SAM:
    def __init__(
        self,
        checkpoint_path: str,
        config: dict = None,
        segmentor_width_size: int = None,
        device: str = None,
    ):
        # 修正模型初始化方式（使用Ultralytics SAM类）
        self.model = UltralyticsSAM(
            model=checkpoint_path
        )
        # 配置推理参数
        self.iou_threshold = config.get("iou_threshold", 0.9)
        self.conf_threshold = config.get("conf_threshold", 0.05)
        self.max_det = config.get("max_det", 200)
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device
        logging.info(f"Init SAM model from {checkpoint_path} done!")

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸以适应模型输入"""
        orig_h, orig_w = image.shape[:2]
        if self.segmentor_width_size is None:
            return image
        # 保持宽高比缩放
        new_w = self.segmentor_width_size
        new_h = int(orig_h * new_w / orig_w)
        return cv2.resize(image, (new_w, new_h))

    def postprocess_resize(self, detections: Dict[str, torch.Tensor], orig_size: tuple) -> Dict[str, torch.Tensor]:
        """修正SAM2的后处理逻辑，完全避免原地操作"""
        if self.segmentor_width_size is None:
            return detections
        
        # 1. 缩放掩码：使用非原地赋值
        masks = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=orig_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        detections["masks"] = masks  # 替换为新张量，非原地修改
        
        # 2. 缩放边界框：彻底避免 *= 等原地运算符
        scale_w = orig_size[1] / self.segmentor_width_size
        scale_h = orig_size[0] / (self.segmentor_width_size * orig_size[0] / orig_size[1])
        
        # 关键修正：先克隆张量，再通过赋值修改（与原有SAM/FastSAM保持一致）
        boxes = detections["boxes"].clone()  # 克隆避免修改原始张量
        # 用普通赋值替代 *=
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h  # y1, y2
        # 裁剪边界时同样使用非原地操作
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_size[1] - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_size[0] - 1)
        detections["boxes"] = boxes  # 替换为新张量
        
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray, prompts: Dict = None) -> List[Dict[str, Any]]:
        """生成图像的实例分割掩码，支持多种提示（边界框/点）"""
        orig_size = image.shape[:2]
        # 预处理：调整尺寸
        resized_image = self.preprocess_resize(image)
        
        # 推理参数（兼容官方API）
        infer_kwargs = {
            "iou": self.iou_threshold,
            "conf": self.conf_threshold,
            "max_det": self.max_det,
            "stream": False
        }
        # 添加提示信息（如边界框、点）
        if prompts is not None:
            infer_kwargs.update(prompts)
        
        # 模型推理（使用Ultralytics SAM API）
        results = self.model(resized_image, **infer_kwargs)
        
        # 提取掩码和边界框
        masks = results[0].masks.data  # 形状: (N, H, W)
        boxes = results[0].boxes.data[:, :4]  # 形状: (N, 4)，XYXY格式
        
        # 后处理：缩放回原图尺寸
        detections = {
            "masks": masks.to(self.current_device),
            "boxes": boxes.to(self.current_device)
        }
        detections = self.postprocess_resize(detections, orig_size)
        return detections
