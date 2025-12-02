from ultralytics import YOLO
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import pytorch_lightning as pl


class CustomYOLO(YOLO):
    def __init__(
        self,
        model,
        iou,
        conf,
        max_det,
        segmentor_width_size,
        selected_device="cpu",
        verbose=False,
    ):
        # 调用父类初始化，设置默认参数
        super().__init__(model=model, verbose=verbose)
        
        # 配置预测参数
        self.iou = iou
        self.conf = conf
        self.max_det = max_det
        self.segmentor_width_size = segmentor_width_size
        self.selected_device = selected_device

        # 设置设备
        self.to(selected_device)
        
        logging.info(f"Init CustomYOLO done!")

    def setup_model(self, device, verbose=False):
        """初始化模型并设置为评估模式"""
        self.to(device)
        self.eval()
        logging.info(f"Setup model at device {device} done!")

    def __call__(self, source=None, stream=False):
        # 使用新版本的predict方法，传递必要参数
        return self.predict(
            source=source,
            stream=stream,
            iou=self.iou,
            conf=self.conf,
            max_det=self.max_det,
            imgsz=self.segmentor_width_size,
            save=False,
            verbose=False
        )


class FastSAM(object):
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: dict = None,
        segmentor_width_size=None,
        device=None,
    ):
        self.model = CustomYOLO(
            model=checkpoint_path,
            iou=config.iou_threshold,
            conf=config.conf_threshold,
            max_det=config.max_det,
            selected_device=device,
            segmentor_width_size=segmentor_width_size,
        )
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device
        logging.info(f"Init FastSAM done!")

    def postprocess_resize(self, detections, orig_size, update_boxes=False):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        if update_boxes:
            scale = orig_size[1] / self.segmentor_width_size
            detections["boxes"] = detections["boxes"].float() * scale
            detections["boxes"][:, [0, 2]] = torch.clamp(
                detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
            )
            detections["boxes"][:, [1, 3]] = torch.clamp(
                detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
            )
        return detections

    @torch.no_grad()
    def generate_masks(self, image) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
        # 调用修改后的预测方法
        detections = self.model(image)

        # 适配新版本的输出格式
        masks = detections[0].masks.data if detections[0].masks is not None else torch.tensor([])
        boxes = detections[0].boxes.xyxy if detections[0].boxes is not None else torch.tensor([])

        # 定义输出数据结构
        mask_data = {
            "masks": masks.to(self.current_device),
            "boxes": boxes.to(self.current_device),
        }
        if self.segmentor_width_size is not None and not mask_data["masks"].numel() == 0:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data