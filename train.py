from multiprocessing import freeze_support
from ultralytics import YOLO

def train():
    model = YOLO(model = "./yolov8n-seg.pt", task = "segmentation")

    model.train(data="./train.yaml", epochs=800, workers=1, batch=20, imgsz=640, patience=0,
                hsv_h = 0,
                hsv_s = 0,
                hsv_v = 0.05,
                )

if __name__ == "__main__":
    freeze_support()
    train()
