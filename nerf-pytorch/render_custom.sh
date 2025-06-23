#!/bin/bash

# 自定义 NeRF 渲染脚本
# 使用方法：
# ./render_custom.sh [轨迹类型] [输出文件名] [模型目录] [其他参数...]

# 默认参数
MODEL_DIR=${3:-"logs/colmap_test"}
PATH_TYPE=${1:-"miku"}
OUTPUT_FILE=${2:-"miku_render.mp4"}
MIKU_JSON=${4:-"miku_camera_path.json"}

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=7

# 创建日志文件名
LOG_FILE="render_${PATH_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "开始自定义 NeRF 渲染..."
echo "使用GPU设备: $CUDA_VISIBLE_DEVICES"
echo "模型目录: $MODEL_DIR"
echo "轨迹类型: $PATH_TYPE"
echo "输出文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "错误：模型目录 '$MODEL_DIR' 不存在！"
    echo "请确保已完成模型训练，或指定正确的模型目录路径"
    exit 1
fi

# 检查 miku 相机文件（如果使用 miku 模式）
if [ "$PATH_TYPE" = "miku" ] && [ ! -f "$MIKU_JSON" ]; then
    echo "错误：找不到相机文件 '$MIKU_JSON'！"
    echo "请确保相机路径文件存在"
    exit 1
fi

# 根据轨迹类型设置不同的参数
case $PATH_TYPE in
    "miku")
        echo "使用 miku 相机路径..."
        echo "开始后台渲染，日志输出到: $LOG_FILE"
        nohup python custom_render.py \
            --model_dir $MODEL_DIR \
            --path_type miku \
            --output $OUTPUT_FILE \
            --miku_json $MIKU_JSON > $LOG_FILE 2>&1 &
        
        # 获取进程ID
        PID=$!
        echo "后台进程ID: $PID"
        echo "使用以下命令查看进度："
        echo "  tail -f $LOG_FILE"
        echo "  ps aux | grep $PID"
        ;;
    "circular")
        echo "使用圆形轨迹..."
        echo "开始后台渲染，日志输出到: $LOG_FILE"
        nohup python custom_render.py \
            --model_dir $MODEL_DIR \
            --path_type circular \
            --output $OUTPUT_FILE \
            --num_frames 120 \
            --radius 4.0 \
            --elevation -30.0 \
            --start_angle 0 \
            --end_angle 360 \
            --fps 30 > $LOG_FILE 2>&1 &
        
        # 获取进程ID
        PID=$!
        echo "后台进程ID: $PID"
        echo "使用以下命令查看进度："
        echo "  tail -f $LOG_FILE"
        echo "  ps aux | grep $PID"
        ;;
    "spiral")
        echo "使用螺旋轨迹..."
        echo "开始后台渲染，日志输出到: $LOG_FILE"
        nohup python custom_render.py \
            --model_dir $MODEL_DIR \
            --path_type spiral \
            --output $OUTPUT_FILE \
            --num_frames 120 \
            --radius 4.0 \
            --elevation -30.0 \
            --start_angle 0 \
            --end_angle 720 \
            --height_variation 2.0 \
            --fps 30 > $LOG_FILE 2>&1 &
        
        # 获取进程ID
        PID=$!
        echo "后台进程ID: $PID"
        echo "使用以下命令查看进度："
        echo "  tail -f $LOG_FILE"
        echo "  ps aux | grep $PID"
        ;;
    "linear")
        echo "使用线性轨迹..."
        echo "开始后台渲染，日志输出到: $LOG_FILE"
        nohup python custom_render.py \
            --model_dir $MODEL_DIR \
            --path_type linear \
            --output $OUTPUT_FILE \
            --num_frames 60 \
            --radius 4.0 \
            --elevation -30.0 \
            --start_angle 0 \
            --end_angle 180 \
            --fps 30 > $LOG_FILE 2>&1 &
        
        # 获取进程ID
        PID=$!
        echo "后台进程ID: $PID"
        echo "使用以下命令查看进度："
        echo "  tail -f $LOG_FILE"
        echo "  ps aux | grep $PID"
        ;;
    "custom")
        echo "使用自定义轨迹..."
        POSES_FILE=${5:-"example_poses.json"}
        echo "开始后台渲染，日志输出到: $LOG_FILE"
        nohup python custom_render.py \
            --model_dir $MODEL_DIR \
            --path_type custom \
            --output $OUTPUT_FILE \
            --poses_file $POSES_FILE \
            --fps 30 > $LOG_FILE 2>&1 &
        
        # 获取进程ID
        PID=$!
        echo "后台进程ID: $PID"
        echo "使用以下命令查看进度："
        echo "  tail -f $LOG_FILE"
        echo "  ps aux | grep $PID"
        ;;
    *)
        echo "错误：不支持的轨迹类型 '$PATH_TYPE'"
        echo "支持的类型：miku, circular, spiral, linear, custom"
        echo ""
        echo "使用方法："
        echo "  ./render_custom.sh miku [输出文件] [模型目录] [相机文件]"
        echo "  ./render_custom.sh circular [输出文件] [模型目录]"
        echo "  ./render_custom.sh spiral [输出文件] [模型目录]"
        echo "  ./render_custom.sh linear [输出文件] [模型目录]"
        echo "  ./render_custom.sh custom [输出文件] [模型目录] [位姿文件]"
        echo ""
        echo "示例："
        echo "  ./render_custom.sh miku my_video.mp4 logs/colmap_test miku_camera_path.json"
        echo "  ./render_custom.sh circular circle.mp4 logs/colmap_test"
        exit 1
        ;;
esac

# 等待一下让后台进程启动
sleep 2

# 检查进程是否还在运行
if ps -p $PID > /dev/null 2>&1; then
    echo ""
    echo "渲染任务已在后台启动！"
    echo "进程ID: $PID"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "监控命令："
    echo "  tail -f $LOG_FILE    # 查看实时日志"
    echo "  ps aux | grep $PID   # 检查进程状态"
    echo "  kill $PID            # 终止进程（如需要）"
    echo ""
    echo "渲染完成后，输出文件将保存为: $OUTPUT_FILE"
else
    echo "渲染进程启动失败！请检查日志文件: $LOG_FILE"
    exit 1
fi 