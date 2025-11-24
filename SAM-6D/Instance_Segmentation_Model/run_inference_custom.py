# 导入必要的库
import os
import sys
import numpy as np
import shutil  # 文件操作工具
from tqdm import tqdm  # 进度条显示
import time
import torch  # 深度学习框架
from PIL import Image  # 图像处理库
import logging  # 日志输出
import os.path as osp  # 路径处理工具
from hydra import initialize, compose  # 配置管理工具
logging.basicConfig(level=logging.INFO)  # 配置日志级别为INFO
import trimesh  # 3D模型处理库
import numpy as np
from hydra.utils import instantiate  # 从配置实例化对象
import argparse  # 命令行参数解析
import glob  # 文件路径匹配
from omegaconf import DictConfig, OmegaConf  # 配置文件处理
from torchvision.utils import save_image  # 保存图像
import torchvision.transforms as T  # 图像变换工具
import cv2  # 计算机视觉库
import imageio.v2 as imageio  # 图像读写
import distinctipy  # 生成distinct颜色的库

# 边缘检测和形态学操作
from skimage.feature import canny  # 边缘检测
from skimage.morphology import binary_dilation  # 二值膨胀操作

# 自定义工具函数导入
from utils.poses.pose_utils import (
    get_obj_poses_from_template_level,  # 从模板级别获取物体姿态
    load_index_level_in_level2  # 加载level2中的索引
)
from utils.bbox_utils import CropResizePad  # 裁剪、缩放、填充工具
from model.utils import Detections, convert_npz_to_json  # 检测结果处理工具
from model.loss import Similarity  # 相似度计算
from utils.inout import (  # 输入输出工具
    load_json,  # 加载json文件
    save_json_bop23,  # 按BOP23格式保存json
    save_torch,  # 保存torch张量
    load_torch  # 加载torch张量
)
from utils.data_utils import rle_to_binary_mask as rle_to_mask  # RLE编码转掩码


# 定义图像反归一化变换：将模型输入的归一化图像恢复为原始RGB范围
# 注：PyTorch中常用的归一化均值为[0.485, 0.456, 0.406]，标准差为[0.229, 0.224, 0.225]
# 反归一化公式为：x = x*std + mean，这里转换为逆操作
inv_rgb_transform = T.Compose(
    [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  # 反归一化均值
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],  # 反归一化标准差
        ),
    ]
)


def visualize(rgb, det, color, save_path="tmp.png"):
    """
    可视化检测结果：将物体掩码和边缘叠加到原图上，并与原图拼接
    
    参数:
        rgb: 原始RGB图像(PIL Image)
        det: 检测结果字典，包含'segmentation'(RLE编码)和'category_id'等信息
        color: 用于标记该物体的颜色(RGB值，0-1范围)
        save_path: 可视化结果保存路径
    返回:
        拼接后的图像(PIL Image)，左侧为原图，右侧为带掩码的结果图
    """
    # 复制原图用于处理
    img = rgb.copy()
    # 转换为灰度图再转RGB(为了后续叠加颜色时更明显)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    alpha = 0.33  # 掩码叠加的透明度

    # 将RLE编码的分割结果转换为二值掩码(0/1)
    mask = rle_to_mask(det["segmentation"])
    # 对掩码进行边缘检测(用于突出显示物体边界)
    edge = canny(mask)
    # 对边缘进行膨胀操作，使边界更清晰
    edge = binary_dilation(edge, np.ones((2, 2)))
    # 获取物体ID和模板ID(模板ID=物体ID-1)
    obj_id = det["category_id"]
    temp_id = obj_id - 1

    # 将颜色值从0-1范围转换为0-255(图像像素范围)
    r = int(255 * color[0])
    g = int(255 * color[1])
    b = int(255 * color[2])
    
    # 将掩码区域与颜色叠加(带透明度)
    img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]  # R通道
    img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]  # G通道
    img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]  # B通道
    # 将边缘设置为白色(255)突出显示
    img[edge, :] = 255
    
    # 保存处理后的图像并重新读取(确保格式正确)
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # 将原图和处理后的结果拼接
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))  # 左侧粘贴原图
    concat.paste(prediction, (img.shape[1], 0))  # 右侧粘贴带掩码的图
    return concat


def batch_input_data(depth_path, cam_path, device):
    """
    处理深度图和相机参数，转换为模型输入的批次数据格式
    
    参数:
        depth_path: 深度图像路径
        cam_path: 相机参数文件(json)路径
        device: 计算设备(cpu/cuda)
    返回:
        batch: 包含预处理后的数据字典，键包括'depth', 'cam_intrinsic', 'depth_scale'
    """
    batch = {}
    # 加载相机参数(内参、深度缩放因子等)
    cam_info = load_json(cam_path)
    # 读取深度图并转换为int32格式
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    # 解析相机内参(3x3矩阵)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    # 获取深度缩放因子(将深度值转换为实际距离)
    depth_scale = np.array(cam_info['depth_scale'])

    # 转换为torch张量，增加批次维度，并移动到目标设备
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh, reset_descriptors=False):
    """
    执行推理流程：初始化模型、处理模板、计算描述符、检测物体、计算分数、保存结果
    
    参数:
        segmentor_model: 分割模型名称(sam/fastsam/sam2/yolov11seg)
        output_dir: 输出结果根目录
        cad_path: CAD模型文件路径(毫米单位)
        rgb_path: RGB图像路径
        depth_path: 深度图像路径(毫米单位)
        cam_path: 相机参数文件路径
        stability_score_thresh: SAM模型的稳定性分数阈值
        reset_descriptors: 是否强制重新计算描述符(不加载已保存的)
    """
    # 创建描述符保存目录(用于存储模板的特征描述符)
    descriptors_dir = osp.join(output_dir, "descriptors")
    os.makedirs(descriptors_dir, exist_ok=True)  # 若目录已存在则不报错
    
    # 定义主描述符和外观描述符的保存路径
    main_desc_path = osp.join(descriptors_dir, "main_descriptors.pth")
    appe_desc_path = osp.join(descriptors_dir, "appe_descriptors.pth")

    # 初始化Hydra配置(加载推理总配置)
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    # 根据选择的分割模型加载对应配置
    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')  # SAM模型配置
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')  # FastSAM模型配置
    elif segmentor_model == "sam2":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam2.yaml')  # SAM2模型配置
    elif segmentor_model == "yolov11seg":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_yolov11seg.yaml')  # YOLOv11分割模型配置
    else:
        raise ValueError(f"不支持的分割模型: {segmentor_model}")

    logging.info("初始化模型")
    # 从配置实例化模型(ISM: Instance Segmentation and Matching)
    model = instantiate(cfg.model)
    
    # 选择计算设备(GPU优先)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将描述符模型移动到目标设备
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    
    # 将分割模型移动到目标设备(不同模型的结构可能不同，需要适配)
    if hasattr(model.segmentor_model, "predictor"):
        # 如SAM模型有predictor属性
        model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
    elif hasattr(model.segmentor_model.model, "to"):
        # 普通PyTorch模型直接调用to方法
        model.segmentor_model.model = model.segmentor_model.model.to(device)
    else:
        # 其他模型可能需要特殊的设备设置方法
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"模型已移动到 {device}")
        
    
    logging.info("初始化模板")
    # 模板目录(存储物体的参考图像和掩码)
    template_dir = os.path.join(output_dir, 'templates')
    # 统计模板数量(通过匹配模板文件路径)
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []  # 存储模板的边界框、掩码、图像
    
    # 加载所有模板数据
    for idx in range(num_templates):
        # 读取模板RGB图像和掩码
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        # 获取掩码的边界框(用于后续裁剪)
        boxes.append(mask.getbbox())

        # 图像预处理：转换为numpy数组并归一化到0-1
        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        # 用掩码过滤图像(只保留物体区域)
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))  # 增加通道维度
        
    # 转换为批次格式：(num_templates, 3, H, W)
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)  # 掩码批次格式：(num_templates, 1, H, W)
    boxes = torch.tensor(np.array(boxes))  # 边界框转换为tensor
    
    # 初始化裁剪缩放处理器(将模板统一处理为224x224)
    processing_config = OmegaConf.create({"image_size": 224})
    proposal_processor = CropResizePad(processing_config.image_size)
    # 对模板图像和掩码进行裁剪缩放，并移动到设备
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    # 加载或计算模板的描述符(特征向量，用于匹配)
    model.ref_data = {}  # 存储参考数据(描述符、姿态等)
    # 主描述符(如CLS token特征)
    if os.path.exists(main_desc_path) and not reset_descriptors:
        logging.info(f"加载主描述符从 {main_desc_path}")
        model.ref_data["descriptors"] = load_torch(main_desc_path).to(device)
    else:
        logging.info("计算主描述符...")
        # 从模板图像计算主描述符(使用cls token特征)
        model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        logging.info(f"保存主描述符到 {main_desc_path}")
        save_torch(model.ref_data["descriptors"], main_desc_path)

    # 外观描述符(掩码区域的patch特征)
    if os.path.exists(appe_desc_path) and not reset_descriptors:
        logging.info(f"加载外观描述符从 {appe_desc_path}")
        model.ref_data["appe_descriptors"] = load_torch(appe_desc_path).to(device)
    else:
        logging.info("计算外观描述符...")
        # 从模板掩码区域计算外观描述符
        model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]  # 取掩码的第一个通道
                    ).unsqueeze(0).data
        logging.info(f"保存外观描述符到 {appe_desc_path}")
        save_torch(model.ref_data["appe_descriptors"], appe_desc_path)
    
    # 运行推理：处理输入图像并检测物体
    # 读取RGB图像
    rgb = Image.open(rgb_path).convert("RGB")
    # 使用分割模型生成掩码(得到初步检测结果)
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    # 封装检测结果(便于后续处理)
    detections = Detections(detections)
    
    # 过滤小目标
    detections.remove_very_small_detections(cfg.model.post_processing_config.mask_post_processing)
    
    # 计算查询图像中检测目标的描述符(主描述符和外观描述符)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

    # 计算语义分数：通过匹配查询描述符与模板描述符，筛选出可能的目标
    (
        idx_selected_proposals,  # 筛选后的检测目标索引
        pred_idx_objects,  # 预测的物体ID
        semantic_score,  # 语义匹配分数
        best_template,  # 最佳匹配的模板索引
    ) = model.compute_semantic_score(query_decriptors)

    # 根据筛选结果更新检测目标
    detections.filter(idx_selected_proposals)
    # 同步更新外观描述符(只保留筛选后的目标)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # 计算外观分数：匹配查询与模板的外观描述符
    appe_scores, ref_aux_descriptor = model.compute_appearance_score(
        best_template, pred_idx_objects, query_appe_descriptors
    )

    # 处理深度和相机数据，准备几何分数计算
    batch = batch_input_data(depth_path, cam_path, device)
    # 获取模板的姿态分布(用于投影计算)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4  # 缩放平移分量(可能是单位转换)
    # 转换姿态为tensor并移动到设备
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    # 加载level2中的姿态索引
    model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

    # 加载CAD模型并采样点云(用于几何匹配)
    mesh = trimesh.load_mesh(cad_path)
    # 采样2048个点，并转换单位为米(输入为毫米，除以1000)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    # 将模板投影到图像平面，得到像素坐标
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    # 计算几何分数：通过点云投影与掩码的匹配度评估
    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
    )

    # 综合分数计算：语义分数 + 外观分数 + 几何分数*可见比例，归一化
    final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)

    # 给检测结果添加分数和物体ID属性
    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))  # 物体ID初始化为0(可根据实际情况修改)
    
    # 执行NMS
    detections.apply_nms(nms_thresh=cfg.model.post_processing_config.nms_thresh)
         
    # 转换检测结果为numpy格式，便于保存
    detections.to_numpy()
    # 定义结果保存路径
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    # 保存检测结果为npz格式
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    # 转换npz结果为json格式(BOP23格式)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    
    # 可视化检测结果(如果有检测结果)
    if detections:
        # 为每个检测目标生成不同的颜色
        colors = distinctipy.get_colors(len(detections))
        
        for idx, det in enumerate(detections):
            # 定义可视化结果保存路径(包含分数)
            vis_save_path = f"{output_dir}/sam6d_results/vis_ism_{idx}_score_{det['score']:.4f}.png"
            # 生成可视化图像
            vis_img = visualize(rgb, det, colors[idx], vis_save_path)
            # 保存可视化结果
            vis_img.save(vis_save_path)
            logging.info(f"已保存检测结果 {idx} 的可视化图片（分数: {det['score']:.4f}）到 {vis_save_path}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="ISM中使用的分割模型(sam/fastsam/sam2/yolov11seg)")
    parser.add_argument("--output_dir", nargs="?", help="输出结果的根目录路径")
    parser.add_argument("--cad_path", nargs="?", help="CAD模型文件路径(毫米单位)")
    parser.add_argument("--rgb_path", nargs="?", help="RGB图像路径")
    parser.add_argument("--depth_path", nargs="?", help="深度图像路径(毫米单位)")
    parser.add_argument("--cam_path", nargs="?", help="相机信息文件路径(json格式)")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="SAM模型的稳定性分数阈值")
    parser.add_argument("--reset_descriptors", action="store_true", help="强制重新计算描述符，不加载已保存的文件")
    args = parser.parse_args()
    
    # 创建结果保存目录
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    
    # 执行推理
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
        reset_descriptors=args.reset_descriptors
    )