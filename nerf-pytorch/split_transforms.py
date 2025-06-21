import os
import json
import random
import re

# 路径设置
original_json_path = 'nerf-pytorch/data/transforms.json'
output_train_json = 'nerf-pytorch/data/transforms_train.json'
output_val_json = 'nerf-pytorch/data/transforms_val.json'
output_test_json = 'nerf-pytorch/data/transforms_test.json'

# 设置随机种子，确保结果可复现
random.seed(42)

# 数据集比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# 读取原始transforms.json
with open(original_json_path, 'r') as f:
    transforms_data = json.load(f)

# 获取所有帧
all_frames = transforms_data['frames']
total_frames = len(all_frames)

# 随机打乱帧的顺序
random.shuffle(all_frames)

# 计算各个集合的大小
train_size = int(total_frames * train_ratio)
val_size = int(total_frames * val_ratio)
# test_size = total_frames - train_size - val_size

# 划分数据集
train_frames = all_frames[:train_size]
val_frames = all_frames[train_size:train_size+val_size]
test_frames = all_frames[train_size+val_size:]

# 创建各个数据集的transforms数据
train_transforms = {k: v for k, v in transforms_data.items() if k != 'frames'}
val_transforms = {k: v for k, v in transforms_data.items() if k != 'frames'}
test_transforms = {k: v for k, v in transforms_data.items() if k != 'frames'}

# 添加frames到各自的transforms数据中
train_transforms['frames'] = train_frames
val_transforms['frames'] = val_frames
test_transforms['frames'] = test_frames

# 保存各个数据集的transforms.json
with open(output_train_json, 'w') as f:
    json.dump(train_transforms, f, indent=4)

with open(output_val_json, 'w') as f:
    json.dump(val_transforms, f, indent=4)

with open(output_test_json, 'w') as f:
    json.dump(test_transforms, f, indent=4)

# 输出统计信息
print(f"总帧数: {total_frames}")
print(f"已生成训练集transforms文件: {output_train_json}, 包含 {len(train_frames)} 帧 ({len(train_frames)/total_frames*100:.1f}%)")
print(f"已生成验证集transforms文件: {output_val_json}, 包含 {len(val_frames)} 帧 ({len(val_frames)/total_frames*100:.1f}%)")
print(f"已生成测试集transforms文件: {output_test_json}, 包含 {len(test_frames)} 帧 ({len(test_frames)/total_frames*100:.1f}%)")

# 输出一些样本帧信息，便于检查
print("\n训练集样本:")
for i in range(min(3, len(train_frames))):
    print(f"  {train_frames[i]['file_path']}")

print("\n验证集样本:")
for i in range(min(3, len(val_frames))):
    print(f"  {val_frames[i]['file_path']}")

print("\n测试集样本:")
for i in range(min(3, len(test_frames))):
    print(f"  {test_frames[i]['file_path']}") 