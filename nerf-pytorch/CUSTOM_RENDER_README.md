# NeRF 自定义相机视角渲染指南

本指南将帮助您使用训练好的 NeRF 模型生成自定义相机视角的渲染视频。

## 概述

训练完成后，NeRF 会在 `logs/colmap_test/` 目录下生成类似 `colmap_test_spiral_050000_rgb.mp4` 的默认渲染视频。现在您可以使用我们提供的工具来生成具有自定义相机轨迹的视频。

**新功能：现在支持 miku_camera_path.json 格式的相机路径文件！**

## 文件说明

- `custom_render.py` - 主要的自定义渲染脚本
- `render_custom.sh` - 便于使用的 bash 脚本
- `miku_camera_path.json` - 您的相机路径文件
- `CUSTOM_RENDER_README.md` - 本使用说明

## 快速开始

### 1. 使用您的 miku 相机路径（推荐）

```bash
# 进入 nerf-pytorch 目录
cd /nvme/liyu/nerf3dgs/nerf-pytorch

# 使用您的相机路径渲染视频
./render_custom.sh miku my_miku_video.mp4 logs/colmap_test miku_camera_path.json
```

### 2. 其他轨迹类型

```bash
# 圆形轨迹
./render_custom.sh circular my_circular_video.mp4 logs/colmap_test

# 螺旋轨迹
./render_custom.sh spiral my_spiral_video.mp4 logs/colmap_test

# 线性轨迹
./render_custom.sh linear my_linear_video.mp4 logs/colmap_test
```

## 详细使用方法

### 使用 miku 相机路径

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type miku \
    --output my_miku_render.mp4 \
    --miku_json miku_camera_path.json
```

### 使用其他轨迹类型

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type circular \
    --output my_render.mp4 \
    --num_frames 120 \
    --radius 4.0 \
    --elevation -30.0 \
    --start_angle 0 \
    --end_angle 360 \
    --fps 30
```

### 参数说明

#### 基本参数
- `--model_dir`: 模型目录路径（如 `logs/colmap_test`，必需）
- `--config`: 配置文件路径（可选，会自动从模型目录查找）
- `--output`: 输出视频文件名（默认：`custom_render.mp4`）
- `--path_type`: 轨迹类型（`miku`, `circular`, `spiral`, `linear`, `custom`）
- `--checkpoint`: 指定检查点路径（可选，会自动选择最新的）

#### miku 模式参数
- `--miku_json`: miku 格式的相机路径文件（默认：`miku_camera_path.json`）

#### 传统轨迹参数
- `--num_frames`: 视频帧数（默认：120，miku模式下会被覆盖）
- `--fps`: 视频帧率（默认：30，miku模式下会被覆盖）
- `--radius`: 相机到场景中心的距离（默认：4.0）
- `--elevation`: 相机仰角，单位度（默认：-30.0）
- `--start_angle`: 起始角度，单位度（默认：0）
- `--end_angle`: 结束角度，单位度（默认：360）
- `--height_variation`: 高度变化（螺旋轨迹用，默认：0.0）

#### 渲染参数
- `--render_factor`: 渲染缩放因子（0=原分辨率，默认：0）

## miku_camera_path.json 格式说明

您的相机路径文件包含以下信息：

```json
{
    "default_fov": 75.0,
    "default_transition_sec": 2.0,
    "fps": 30.0,
    "seconds": 30.0,
    "keyframes": [
        {
            "matrix": [16个数字组成的4x4变换矩阵],
            "fov": 75.0,
            "aspect": 1.7777777777777777
        },
        ...
    ],
    "camera_path": [
        {
            "camera_to_world": [16个数字组成的4x4变换矩阵],
            "fov": 75.0,
            "aspect": 1.7777777777777777
        },
        ...
    ]
}
```

脚本会自动：
- 从 `fps` 和 `seconds` 计算总帧数
- 从 `keyframes` 或 `camera_path` 读取相机位姿
- 如果关键帧数量不足，进行插值补间
- 如果关键帧过多，进行采样

## 轨迹类型详解

#### 1. miku 模式 (推荐)
使用您提供的 miku_camera_path.json 文件中的相机轨迹。

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type miku \
    --miku_json miku_camera_path.json
```

#### 2. 圆形轨迹 (circular)
相机围绕场景中心做圆周运动，保持固定的仰角和距离。

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type circular \
    --radius 4.0 \
    --elevation -30.0 \
    --start_angle 0 \
    --end_angle 360 \
    --num_frames 120
```

#### 3. 螺旋轨迹 (spiral)
相机围绕场景中心做螺旋运动，同时改变仰角。

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type spiral \
    --radius 4.0 \
    --elevation -30.0 \
    --start_angle 0 \
    --end_angle 720 \
    --height_variation 2.0 \
    --num_frames 120
```

#### 4. 线性轨迹 (linear)
相机在两个位置之间做线性插值运动。

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type linear \
    --start_angle 0 \
    --end_angle 180 \
    --num_frames 60
```

#### 5. 自定义轨迹 (custom)
使用简化格式的 JSON 文件定义相机位姿。

```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type custom \
    --poses_file my_poses.json
```

## 示例用法

### 1. 使用您的相机路径（主要用法）
```bash
# 使用默认设置
./render_custom.sh miku

# 指定输出文件名
./render_custom.sh miku my_video.mp4

# 指定模型目录和相机文件
./render_custom.sh miku my_video.mp4 logs/colmap_test miku_camera_path.json
```

### 2. 快速预览（低分辨率）
```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type miku \
    --output preview.mp4 \
    --render_factor 4 \
    --miku_json miku_camera_path.json
```

### 3. 高质量渲染
```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --path_type miku \
    --output high_quality.mp4 \
    --render_factor 0 \
    --miku_json miku_camera_path.json
```

### 4. 使用不同的模型检查点
```bash
python custom_render.py \
    --model_dir logs/colmap_test \
    --checkpoint logs/colmap_test/080000.tar \
    --path_type miku \
    --output specific_checkpoint.mp4
```

## 故障排除

### 常见问题

1. **找不到模型文件**
   ```
   错误：模型目录 'logs/colmap_test' 不存在！
   ```
   解决：确保您已完成模型训练，或指定正确的模型目录路径。

2. **找不到相机文件**
   ```
   错误：找不到相机文件 'miku_camera_path.json'！
   ```
   解决：确保相机路径文件在当前目录下，或使用 `--miku_json` 指定正确路径。

3. **配置文件问题**
   ```
   错误：无法找到配置文件
   ```
   解决：脚本会自动在模型目录中查找 `config.txt` 或 `args.txt`，如果没有找到，请手动指定 `--config` 参数。

4. **内存不足**
   ```
   CUDA out of memory
   ```
   解决：使用 `--render_factor 2` 或 `--render_factor 4` 降低渲染分辨率。

### 性能优化

- **快速预览**：使用 `--render_factor 4` 将分辨率降低到 1/4
- **高质量渲染**：使用 `--render_factor 0`（原分辨率）
- **GPU 内存优化**：如果显存不足，可以尝试 `--render_factor 2`

## 技术细节

### 坐标系转换
脚本会自动处理 miku 格式的相机坐标系到 NeRF 坐标系的转换。

### 插值算法
当关键帧数量少于目标帧数时，脚本使用线性插值和旋转矩阵正交化来生成平滑的相机运动。

### 支持的数据集类型
- Blender 格式数据集
- LLFF 格式数据集
- 其他兼容格式（自动检测）

---

现在您可以使用以下命令开始渲染：

```bash
# 最简单的用法
./render_custom.sh miku

# 或者直接使用 Python
python custom_render.py --model_dir logs/colmap_test --path_type miku
``` 