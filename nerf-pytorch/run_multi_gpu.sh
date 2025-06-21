#!/bin/bash
# 多GPU训练启动脚本

# 使用所有可用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 
# 运行训练脚本
python run_nerf.py --config configs/colmap.txt 