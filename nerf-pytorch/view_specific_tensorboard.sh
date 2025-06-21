#!/bin/bash
# 用法: ./view_specific_tensorboard.sh <experiment_name>
# 例如: ./view_specific_tensorboard.sh colmap_test

if [ -z "$1" ]; then
  echo "请提供实验名称作为参数"
  echo "用法: ./view_specific_tensorboard.sh <experiment_name>"
  exit 1
fi

EXPERIMENT_NAME=$1
LOGDIR="./logs/tensorboard/${EXPERIMENT_NAME}"

if [ ! -d "$LOGDIR" ]; then
  echo "错误: 日志目录 $LOGDIR 不存在"
  echo "可用的实验日志目录:"
  ls -la ./logs/tensorboard/
  exit 1
fi

echo "启动TensorBoard查看实验: $EXPERIMENT_NAME"
echo "日志目录: $LOGDIR"

tensorboard --logdir=$LOGDIR --bind_all 