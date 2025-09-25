from ultralytics import YOLO

model = YOLO('D:/YOLO/ultralytics-v11/ultralytics-8.3.95/ultralytics/cfg/models/11/yolo11-seg-GAM.yaml') # 构建你的 YOLO 模型对象（如 YOLO('你的yaml路径')）
print(model)