from ultralytics import YOLO

MODEL_YOLU = r'C:\Users\alpcu\Desktop\DerinOgrenme\YOLOv8\runs\kirikyertespiti_hizli4\weights\best.pt'

GÖRÜNTÜ_YOLU = 'kirikyok-2.png' 

model = YOLO(MODEL_YOLU)

print(f"Model yükleniyor: {MODEL_YOLU}")
print(f"Tahmin görüntüsü: {GÖRÜNTÜ_YOLU}")

results = model.predict(source=GÖRÜNTÜ_YOLU, 
                        conf=0.5, #modelin işaretlemeden duyduğu güven
                        save=True, 
                        device=0)


print("\nTESPİT TAMAMLANDI!")
print("Sonuçlar 'runs/detect/predict/' klasöründe kaydedildi.") 
