from ultralytics import YOLO
import torch

# 1. 构建模型（用你的yaml或权重路径）
model = YOLO('D:/YOLO/ultralytics-v11/ultralytics-8.3.95/ultralytics/cfg/models/11/yolo11-seg-GAM.yaml')

# 2. 准备假输入
dummy_input = torch.randn(1, 3, 640, 640)  # 根据模型实际输入尺寸调整

# 3. Forward推理
out = model(dummy_input)
print('Forward output:', out)