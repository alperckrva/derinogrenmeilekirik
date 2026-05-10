import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import cv2

# Model yapısını tanımla (DenseNet121)
def load_densenet_model(model_path, device, num_classes=2):
    """DenseNet121 modelini yükle"""
    
    # Pretrained DenseNet121 yapısını oluştur
    weight = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weight)
    
    # Classifier'ı değiştir (eğitim sırasında yapıldığı gibi)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    # Eğitilmiş ağırlıkları yükle
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    return model

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}\n")

# Model yükle
MODEL_PATH = r"C:\Users\alpcu\Desktop\DerinOgrenme\Kirik var mi yok mu\DenseNet121_416x416\best_model.pth"
print("Model yükleniyor...")
model = load_densenet_model(MODEL_PATH, device)
print("✅ Model yüklendi!\n")

# Transform (eğitim sırasında kullanılan ile aynı)
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fotoğraflar
image_paths = ["1.png", "2.png", "3.png"]

print("="*60)
print("KIRIK TESPİT SİSTEMİ (VAR/YOK)")
print("="*60)

os.makedirs("results", exist_ok=True)

for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"⚠️  {img_path} bulunamadı!")
        continue
    
    print(f"\n📸 {img_path}")
    print("-"*60)
    
    # Görüntüyü yükle
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tahmin
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item() * 100
    
    # Sonuç
    class_names = ["KIRIK YOK", "KIRIK VAR"]
    prediction = class_names[pred_class]
    
    icon = "✅" if pred_class == 0 else "❌"
    print(f"{icon} Sonuç: {prediction}")
    print(f"   Güven: %{confidence:.2f}")
    
    # Görselleştir
    img_cv = cv2.imread(img_path)
    h, w = img_cv.shape[:2]
    
    color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)
    cv2.rectangle(img_cv, (10, 10), (w-10, 120), color, -1)
    cv2.putText(img_cv, prediction, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img_cv, f"Guven: %{confidence:.2f}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Kaydet
    output_name = f"{os.path.splitext(img_path)[0]}_processed.png"
    output_path = os.path.join("results", output_name)
    cv2.imwrite(output_path, img_cv)
    print(f"   💾 {output_path}")

print("\n" + "="*60)
print("✨ TAMAMLANDI!")
print("="*60)
