from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")

results = model("D:/YOLO/ultralytics-v11/ultralytics-8.3.95/ultralytics/assets/")