import blenderproc as bproc  # 导入BlenderProc主库，用于程序化控制Blender进行渲染
import os  # 提供操作系统相关功能，如路径操作
import argparse  # 用于解析命令行参数
import cv2  # OpenCV库，用于图像处理操作
import numpy as np  # NumPy库，用于数值计算和数组操作
import trimesh  # 用于3D网格模型的加载和处理

# 创建命令行参数解析器，方便用户配置渲染参数
parser = argparse.ArgumentParser()
# 添加CAD模型路径参数，这是要渲染的3D模型文件
parser.add_argument('--cad_path', help="The path of CAD model")
# 添加输出目录参数，指定渲染结果保存的位置
parser.add_argument('--output_dir', help="The path to save CAD templates")
# 添加归一化选项，决定是否将模型缩放到统一尺度（默认为True）
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
# 添加着色选项，决定是否为模型添加基础材质颜色（默认为False）
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
# 添加基础颜色参数，当colorize为True时使用此颜色（默认为0.05，接近黑色的灰度）
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
# 解析命令行输入的参数
args = parser.parse_args()

# 获取当前脚本文件的绝对路径，用于构建相对路径
render_dir = os.path.dirname(os.path.abspath(__file__))
# 构建预定义相机姿态文件的路径，这些姿态通常围绕物体均匀分布[2]
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

# 初始化BlenderProc环境，创建空的Blender场景[2]
bproc.init()

def get_norm_info(mesh_path):
    """计算模型的归一化尺度因子，使模型能够适应标准化的渲染环境[3]
    
    Args:
        mesh_path: 3D模型文件的路径
        
    Returns:
        float: 缩放因子，用于将模型 bounding box 限制在单位球内
    """
    # 使用trimesh加载网格模型，force参数确保作为网格处理
    mesh = trimesh.load(mesh_path, force='mesh')

    # 从模型表面均匀采样1024个点，用于计算模型尺度
    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    # 计算点云在x、y、z方向上的最小和最大值
    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    # 计算模型的半径（从原点到最远点的距离）
    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    # 返回缩放因子，使模型能适应-0.5到0.5的边界框范围
    return 1/(2*radius)

# 加载预定义的相机姿态数组，这些姿态定义了从不同角度查看物体的方式[2]
cam_poses = np.load(cnos_cam_fpath)

# 根据用户选择决定是否对模型进行归一化缩放
if args.normalize:
    # 计算归一化所需的缩放因子
    scale = get_norm_info(args.cad_path)
else:
    # 不使用归一化时，缩放因子设为1（保持原大小）
    scale = 1

# 遍历所有相机姿态，为每个姿态渲染一组图像
for idx, cam_pose in enumerate(cam_poses):
    # 清空当前Blender场景中的所有对象，为加载新模型做准备[2]
    bproc.clean_up()

    # 加载OBJ格式的CAD模型，返回对象列表并取第一个对象[2]
    obj = bproc.loader.load_obj(args.cad_path)[0]
    # 对模型应用缩放因子，确保所有模型尺寸一致
    obj.set_scale([scale, scale, scale])
    # 设置对象的类别ID，用于后续识别或分割任务[2]
    obj.set_cp("category_id", 1)

    # 如果用户选择为模型着色，则创建并应用基础材质
    if args.colorize:
        # 创建RGBA颜色数组，A通道(透明度)为0表示完全不透明
        color = [args.base_color, args.base_color, args.base_color, 0.]
        # 创建新的材质对象[5]
        material = bproc.material.create('obj')
        # 设置材质的基础颜色参数[5]
        material.set_principled_shader_value('Base Color', color)
        # 将材质应用到对象的第一个材质槽
        obj.set_material(0, material)

    # 调整相机姿态矩阵以适应Blender的坐标系[2]
    # 将y和z轴方向取反，因为Blender与计算机视觉常用坐标系不同
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
    # 将平移向量从毫米转换为米（Blender使用米为单位），并稍微调整尺度
    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
    # 将调整后的相机姿态添加到Blender场景中[2]
    bproc.camera.add_camera_pose(cam_pose)
    
    # 设置场景光照，确保物体被充分照亮
    light_scale = 2.5  # 光源位置相对于相机位置的缩放因子
    light_energy = 1000  # 光照强度值
    # 创建新的光源对象[2]
    light1 = bproc.types.Light()
    # 设置光源类型为点光源（模拟灯泡）[2]
    light1.set_type("POINT")
    # 将光源放置在相机位置附近但稍远的位置，确保物体被照亮
    light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    # 设置光源的能量（亮度）[6]
    light1.set_energy(light_energy)

    # 设置渲染器的最大采样数，控制渲染质量和速度的平衡[6]
    bproc.renderer.set_max_amount_of_samples(50)
    # 执行渲染管线，获取颜色、深度等基本渲染数据[2,6]
    data = bproc.renderer.render()
    # 渲染NOCS图（标准化物体坐标空间），并更新到数据字典中[1]
    data.update(bproc.renderer.render_nocs())
    
    # 检查保存模板的目录是否存在，不存在则创建
    save_fpath = os.path.join(args.output_dir, "templates")
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    # 保存RGB彩色图像（将OpenGL的RGB格式转换为OpenCV的BGR格式）
    color_bgr_0 = data["colors"][0]  # 获取第一张渲染的颜色图像
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]  # RGB转BGR
    # 使用OpenCV将图像保存为PNG格式
    cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

    # 保存掩码图像（从NOCS图的Alpha通道提取）
    mask_0 = data["nocs"][0][..., -1]  # 提取NOCS的Alpha通道作为掩码
    # 将掩码值从[0,1]范围转换为[0,255]并保存
    cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0 * 255)
    
    # 保存NOCS坐标数据（标准化物体坐标空间）
    xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)  # 将[0,1]范围映射到[-1,1]范围
    # 将NOCS坐标保存为NumPy二进制文件，使用float16节省空间
    np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))