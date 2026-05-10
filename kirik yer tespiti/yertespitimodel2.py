from ultralytics import YOLO
import os


def train_yolov8():
    model=YOLO('yolov8s.pt')

    results=model.train(data='GRAZPEDWRI-DX_dataset/fracture_only.yaml',
                        epochs=40,
                        batch=16,          
                        name='kirikyertespiti_hizli4', 
                        device=0,          
                        imgsz=640,         
                        lr0=0.001,
                        optimizer='AdamW',
                        patience=10, 
                        project=r'C:\Users\alpcu\Desktop\DerinOgrenme\YOLOv8\runs',
                        cache=False, 
                        workers=4)         

    print("Eğitim Tamamlandı")

if __name__ == "__main__":
    train_yolov8()