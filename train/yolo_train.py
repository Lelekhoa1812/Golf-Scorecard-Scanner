from ultralytics import YOLO

def train_yolo(data_path="train/data.yaml", model="yolov8x-obb.pt", epochs=50, imgsz=640):
    """Train YOLOv8 OBB model."""
    model = YOLO(model)
    model.train(data=data_path, epochs=epochs, imgsz=imgsz)

if __name__ == "__main__":
    train_yolo()
