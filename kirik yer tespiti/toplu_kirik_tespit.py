from ultralytics import YOLO
import cv2
import os
import shutil

def detect_cracks_batch(image_paths, model_path, output_dir='processed_results'):
    """
    Birden fazla fotoğrafta kırık tespiti yapar ve işaretli sonuçları kaydeder
    
    Args:
        image_paths: Test edilecek fotoğrafların listesi
        model_path: Eğitilmiş model ağırlıklarının yolu
        output_dir: Sonuçların kaydedileceği klasör
    """
    
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Modeli yükle
    print(f"Model yükleniyor: {model_path}")
    model = YOLO(model_path)
    print("✅ Model yüklendi!\n")
    
    # Her fotoğrafı işle
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"⚠️  Fotoğraf bulunamadı: {image_path}")
            continue
        
        # Dosya adını al (1.png, 2.png, 3.png)
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        print("="*60)
        print(f"📸 İşleniyor: {filename}")
        print("="*60)
        
        # Tahmin yap
        results = model.predict(
            source=image_path,
            conf=0.25,  # Güven eşiği
            save=True,  # Sonucu kaydet
            project='temp_results',  # Geçici klasör
            name='detection',
            exist_ok=True
        )
        
        result = results[0]
        
        # Sonuçları göster
        if len(result.boxes) == 0:
            print("❌ Kırık tespit edilemedi!")
        else:
            print(f"✅ {len(result.boxes)} adet kırık tespit edildi!\n")
            
            for i, box in enumerate(result.boxes):
                # Koordinatlar
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Güven skoru
                confidence = box.conf[0].cpu().numpy()
                
                # Sınıf
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                print(f"  Kırık #{i+1}:")
                print(f"    - Konum: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
                print(f"    - Güven: %{confidence*100:.2f}")
                print(f"    - Sınıf: {class_name}")
        
        # İşlenmiş fotoğrafı yeniden adlandır ve kaydet
        temp_result_path = os.path.join('temp_results', 'detection', filename)
        output_filename = f"{name_without_ext}_processed.png"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(temp_result_path):
            shutil.copy(temp_result_path, output_path)
            print(f"\n💾 Kaydedildi: {output_path}")
        else:
            print(f"\n⚠️  Sonuç dosyası bulunamadı!")
        
        print()
    
    # Geçici klasörü temizle
    if os.path.exists('temp_results'):
        shutil.rmtree('temp_results')
    
    print("="*60)
    print("✨ TÜM FOTOĞRAFLAR İŞLENDİ!")
    print(f"📁 Sonuçlar: {os.path.abspath(output_dir)}")
    print("="*60)


if __name__ == "__main__":
    # Model yolu
    MODEL_PATH = r"runs\kirikyertespiti_hizli3\weights\best.pt"
    
    # İşlenecek fotoğraflar
    IMAGE_PATHS = [
        "1.png",
        "2.png",
        "3.png"
    ]
    
    # Kırık tespiti yap
    detect_cracks_batch(IMAGE_PATHS, MODEL_PATH)
