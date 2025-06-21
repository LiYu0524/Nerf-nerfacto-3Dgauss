#!/bin/bash
# 用法: ./view_exact_tensorboard_file.sh <完整的日志文件路径>
# 例如: ./view_exact_tensorboard_file.sh /nvme/liyu/nerf-pytorch/logs/tensorboard/colmap_test/events.out.tfevents.1750087167.nvidia.1861663.0

if [ -z "$1" ]; then
  echo "请提供完整的日志文件路径作为参数"
  echo "用法: ./view_exact_tensorboard_file.sh <完整的日志文件路径>"
  exit 1
fi

LOG_FILE=$1

if [ ! -f "$LOG_FILE" ]; then
  echo "错误: 日志文件 $LOG_FILE 不存在"
  exit 1
fi

# 获取日志文件所在的目录
LOG_DIR=$(dirname "$LOG_FILE")
echo "启动TensorBoard查看日志文件: $(basename $LOG_FILE)"
echo "日志目录: $LOG_DIR"

# 创建临时目录
TEMP_DIR="/tmp/tensorboard_$(date +%s)"
mkdir -p $TEMP_DIR

# 创建符号链接
ln -s "$LOG_FILE" "$TEMP_DIR/$(basename $LOG_FILE)"

echo "使用临时目录: $TEMP_DIR"
tensorboard --logdir=$TEMP_DIR --bind_all

# 清理临时目录
rm -rf $TEMP_DIR 