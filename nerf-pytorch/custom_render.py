#!/usr/bin/env python3

import os
import torch
import numpy as np
import imageio
import json
import argparse
from tqdm import tqdm
import time

# 导入 NeRF 相关函数
from run_nerf import config_parser, create_nerf, render_path, render
from load_blender import pose_spherical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_miku_camera_path(json_file):
    """
    加载 miku_camera_path.json 格式的相机数据
    
    Args:
        json_file: JSON 文件路径
    
    Returns:
        render_poses: 渲染位姿张量
        fps: 帧率
        num_frames: 帧数
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取基本参数
    fps = data.get('fps', 30.0)
    seconds = data.get('seconds', 30.0)
    num_frames = int(fps * seconds)
    
    # 从 keyframes 或 camera_path 获取相机位姿
    if 'camera_path' in data:
        poses_data = data['camera_path']
    elif 'keyframes' in data:
        poses_data = data['keyframes']
    else:
        raise ValueError("JSON 文件中没有找到 'camera_path' 或 'keyframes' 字段")
    
    render_poses = []
    
    for pose_data in poses_data:
        if 'camera_to_world' in pose_data:
            # 使用 camera_to_world 矩阵
            matrix = pose_data['camera_to_world']
        elif 'matrix' in pose_data:
            # 使用 matrix 字段
            matrix = pose_data['matrix']
        else:
            raise ValueError("位姿数据中没有找到变换矩阵")
        
        # 转换为 4x4 矩阵
        if len(matrix) == 16:
            c2w = np.array(matrix).reshape(4, 4)
        else:
            raise ValueError(f"变换矩阵应为16个元素，实际为{len(matrix)}个")
        
        render_poses.append(c2w)
    
    # 如果 keyframes 数量少于目标帧数，进行插值
    if len(render_poses) < num_frames:
        print(f"关键帧数量 ({len(render_poses)}) 少于目标帧数 ({num_frames})，进行插值...")
        render_poses = interpolate_poses(render_poses, num_frames)
    elif len(render_poses) > num_frames:
        # 如果关键帧过多，进行采样
        print(f"关键帧数量 ({len(render_poses)}) 多于目标帧数 ({num_frames})，进行采样...")
        indices = np.linspace(0, len(render_poses)-1, num_frames, dtype=int)
        render_poses = [render_poses[i] for i in indices]
    
    render_poses = torch.stack([torch.tensor(pose, dtype=torch.float32) for pose in render_poses])
    
    return render_poses, fps, num_frames

def interpolate_poses(poses, target_frames):
    """
    对位姿进行插值以达到目标帧数
    
    Args:
        poses: 原始位姿列表
        target_frames: 目标帧数
    
    Returns:
        interpolated_poses: 插值后的位姿列表
    """
    if len(poses) < 2:
        # 如果只有一个位姿，复制多份
        return poses * target_frames
    
    interpolated_poses = []
    
    # 计算插值参数
    t_values = np.linspace(0, len(poses) - 1, target_frames)
    
    for t in t_values:
        # 找到相邻的两个关键帧
        idx = int(np.floor(t))
        if idx >= len(poses) - 1:
            interpolated_poses.append(poses[-1])
            continue
        
        # 插值权重
        alpha = t - idx
        
        # 对位置进行线性插值
        pos1 = poses[idx][:3, 3]
        pos2 = poses[idx + 1][:3, 3]
        interp_pos = (1 - alpha) * pos1 + alpha * pos2
        
        # 对旋转矩阵进行插值（简单的线性插值）
        rot1 = poses[idx][:3, :3]
        rot2 = poses[idx + 1][:3, :3]
        interp_rot = (1 - alpha) * rot1 + alpha * rot2
        
        # 重新正交化旋转矩阵
        u, s, vh = np.linalg.svd(interp_rot)
        interp_rot = u @ vh
        
        # 构建插值后的位姿矩阵
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_rot
        interp_pose[:3, 3] = interp_pos
        
        interpolated_poses.append(interp_pose)
    
    return interpolated_poses

def create_custom_camera_path(path_type='circular', num_frames=120, radius=4.0, height=0.0, 
                             start_angle=0, end_angle=360, elevation=-30.0,
                             custom_poses=None, miku_json_file=None):
    """
    创建自定义相机轨迹
    
    Args:
        path_type: 轨迹类型 ('circular', 'spiral', 'linear', 'custom', 'miku')
        num_frames: 帧数
        radius: 半径
        height: 高度变化
        start_angle: 起始角度
        end_angle: 结束角度  
        elevation: 仰角
        custom_poses: 自定义位姿列表 [(x,y,z,theta,phi), ...]
        miku_json_file: miku 格式的 JSON 文件路径
    
    Returns:
        render_poses: 渲染位姿张量
        fps: 帧率（如果是 miku 格式）
        num_frames: 实际帧数（如果是 miku 格式）
    """
    
    if path_type == 'miku' and miku_json_file is not None:
        # 使用 miku 格式的相机路径
        return load_miku_camera_path(miku_json_file)
    
    elif path_type == 'custom' and custom_poses is not None:
        # 使用自定义位姿
        render_poses = []
        for pose in custom_poses:
            if len(pose) == 5:  # (x, y, z, theta, phi)
                x, y, z, theta, phi = pose
                # 从位置和角度创建位姿矩阵
                c2w = create_pose_from_position_angles(x, y, z, theta, phi)
            elif len(pose) == 16:  # 4x4 变换矩阵
                c2w = np.array(pose).reshape(4, 4)
            else:
                raise ValueError("自定义位姿格式错误，应为 (x,y,z,theta,phi) 或 4x4 矩阵")
            render_poses.append(c2w)
        render_poses = torch.stack([torch.tensor(pose, dtype=torch.float32) for pose in render_poses])
        return render_poses, 30, len(render_poses)  # 默认返回值
        
    elif path_type == 'circular':
        # 圆形轨迹
        angles = np.linspace(start_angle, end_angle, num_frames, endpoint=False)
        render_poses = torch.stack([
            pose_spherical(angle, elevation, radius) 
            for angle in angles
        ], 0)
        return render_poses, 30, num_frames
        
    elif path_type == 'spiral':
        # 螺旋轨迹
        angles = np.linspace(start_angle, end_angle, num_frames, endpoint=False)
        heights = np.linspace(0, height, num_frames)
        render_poses = []
        for i, angle in enumerate(angles):
            current_elevation = elevation + heights[i] * 10  # 调整高度变化
            pose = pose_spherical(angle, current_elevation, radius)
            render_poses.append(pose)
        render_poses = torch.stack(render_poses, 0)
        return render_poses, 30, num_frames
        
    elif path_type == 'linear':
        # 线性轨迹
        start_pose = pose_spherical(start_angle, elevation, radius)
        end_pose = pose_spherical(end_angle, elevation, radius)
        render_poses = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            # 线性插值位置
            interp_pose = (1 - t) * start_pose + t * end_pose
            render_poses.append(interp_pose)
        render_poses = torch.stack(render_poses, 0)
        return render_poses, 30, num_frames
        
    else:
        raise ValueError(f"不支持的轨迹类型: {path_type}")

def create_pose_from_position_angles(x, y, z, theta, phi):
    """
    从位置和角度创建4x4位姿矩阵
    
    Args:
        x, y, z: 相机位置
        theta: 水平角度（度）
        phi: 垂直角度（度）
    
    Returns:
        4x4 位姿矩阵
    """
    # 转换为弧度
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    # 计算相机朝向
    direction = np.array([
        np.cos(phi_rad) * np.cos(theta_rad),
        np.cos(phi_rad) * np.sin(theta_rad),
        np.sin(phi_rad)
    ])
    
    # 计算上向量
    up = np.array([0, 0, 1])
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, direction)
    up = up / np.linalg.norm(up)
    
    # 构建旋转矩阵
    rotation = np.column_stack([right, up, -direction])
    
    # 构建4x4变换矩阵
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    
    return pose

def load_trained_model(config_path, checkpoint_path=None, model_dir=None):
    """
    加载训练好的模型
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点路径（可选，如果不提供则使用最新的）
        model_dir: 模型目录路径（如 logs/colmap_test）
    
    Returns:
        render_kwargs_test: 渲染参数
        hwf: 图像尺寸和焦距
        K: 相机内参矩阵
    """
    # 解析配置
    parser = config_parser()
    
    # 如果提供了模型目录，修改配置文件路径
    if model_dir:
        config_args = ['--config', config_path, '--basedir', model_dir]
        if not checkpoint_path:
            # 自动寻找最新的检查点
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.tar')]
            if ckpt_files:
                # 按文件名排序，取最新的
                ckpt_files.sort()
                checkpoint_path = os.path.join(model_dir, ckpt_files[-1])
                print(f"自动选择检查点: {checkpoint_path}")
    else:
        config_args = ['--config', config_path]
    
    if checkpoint_path:
        config_args.extend(['--ft_path', checkpoint_path])
    
    args = parser.parse_args(config_args)
    
    # 设置为仅渲染模式
    args.render_only = True
    
    # 加载数据（用于获取相机参数）
    if args.dataset_type == 'blender':
        from load_blender import load_blender_data
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        near, far = 2., 6.
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    elif args.dataset_type == 'llff':
        from load_llff import load_llff_data
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        print(f"数据集类型: {args.dataset_type}")
        # 尝试支持更多数据集类型
        from load_llff import load_llff_data
        try:
            images, poses, bds, render_poses, i_test = load_llff_data(
                args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        except:
            raise NotImplementedError(f"不支持的数据集类型: {args.dataset_type}")
    
    # 设置相机内参
    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    
    # 创建模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    
    # 设置边界
    bds_dict = {'near': near, 'far': far}
    render_kwargs_test.update(bds_dict)
    
    return render_kwargs_test, hwf, K, args

def main():
    parser = argparse.ArgumentParser(description='自定义 NeRF 渲染')
    parser.add_argument('--config', type=str, help='配置文件路径（可选，如果提供model_dir会自动查找）')
    parser.add_argument('--model_dir', type=str, help='模型目录路径（如 logs/colmap_test）')
    parser.add_argument('--checkpoint', type=str, help='检查点路径（可选）')
    parser.add_argument('--output', type=str, default='custom_render.mp4', help='输出视频文件名')
    parser.add_argument('--path_type', type=str, default='miku', 
                       choices=['circular', 'spiral', 'linear', 'custom', 'miku'],
                       help='相机轨迹类型')
    parser.add_argument('--num_frames', type=int, default=120, help='帧数（miku模式下会被覆盖）')
    parser.add_argument('--radius', type=float, default=4.0, help='相机距离')
    parser.add_argument('--elevation', type=float, default=-30.0, help='仰角（度）')
    parser.add_argument('--start_angle', type=float, default=0, help='起始角度（度）')
    parser.add_argument('--end_angle', type=float, default=360, help='结束角度（度）')
    parser.add_argument('--height_variation', type=float, default=0.0, help='高度变化（螺旋轨迹用）')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率（miku模式下会被覆盖）')
    parser.add_argument('--render_factor', type=int, default=0, help='渲染缩放因子（0=原分辨率）')
    parser.add_argument('--poses_file', type=str, help='自定义位姿文件（JSON格式）')
    parser.add_argument('--miku_json', type=str, default='miku_camera_path.json', help='miku格式的相机路径文件')
    
    args = parser.parse_args()
    
    # 如果没有提供config但提供了model_dir，尝试自动查找配置文件
    if not args.config and args.model_dir:
        possible_configs = [
            os.path.join(args.model_dir, 'config.txt'),
            os.path.join(args.model_dir, 'args.txt'),
            'configs/colmap.txt'
        ]
        for config_path in possible_configs:
            if os.path.exists(config_path):
                args.config = config_path
                print(f"自动选择配置文件: {config_path}")
                break
        
        if not args.config:
            raise ValueError("无法找到配置文件，请手动指定 --config 参数")
    
    print(f"配置文件: {args.config}")
    print(f"模型目录: {args.model_dir}")
    print(f"轨迹类型: {args.path_type}")
    print(f"输出文件: {args.output}")
    
    # 加载训练好的模型
    print("加载模型...")
    render_kwargs_test, hwf, K, model_args = load_trained_model(
        args.config, args.checkpoint, args.model_dir)
    
    # 创建自定义相机轨迹
    print("创建相机轨迹...")
    custom_poses = None
    miku_json_file = None
    
    if args.path_type == 'miku':
        miku_json_file = args.miku_json
        if not os.path.exists(miku_json_file):
            raise FileNotFoundError(f"找不到 miku 相机文件: {miku_json_file}")
    elif args.path_type == 'custom' and args.poses_file:
        with open(args.poses_file, 'r') as f:
            custom_poses = json.load(f)
    
    result = create_custom_camera_path(
        path_type=args.path_type,
        num_frames=args.num_frames,
        radius=args.radius,
        height=args.height_variation,
        start_angle=args.start_angle,
        end_angle=args.end_angle,
        elevation=args.elevation,
        custom_poses=custom_poses,
        miku_json_file=miku_json_file
    )
    
    if len(result) == 3:
        render_poses, fps, num_frames = result
        args.fps = fps
        args.num_frames = num_frames
    else:
        render_poses = result
    
    # 移动到GPU
    render_poses = render_poses.to(device)
    
    # 确保相机内参也在正确的设备上
    K = torch.tensor(K, dtype=torch.float32).to(device)
    
    print(f"相机轨迹形状: {render_poses.shape}")
    print(f"帧率: {args.fps}")
    print(f"总帧数: {render_poses.shape[0]}")
    
    # 渲染
    print("开始渲染...")
    with torch.no_grad():
        rgbs, disps = render_path(render_poses, hwf, K, model_args.chunk, 
                                 render_kwargs_test, render_factor=args.render_factor)
    
    print(f"渲染完成，生成 {len(rgbs)} 帧")
    
    # 保存视频
    print(f"保存视频到 {args.output}...")
    def to8b(x): 
        return (255*np.clip(x,0,1)).astype(np.uint8)
    
    imageio.mimwrite(args.output, to8b(rgbs), fps=args.fps, quality=8)
    print("完成！")

if __name__ == '__main__':
    main() 