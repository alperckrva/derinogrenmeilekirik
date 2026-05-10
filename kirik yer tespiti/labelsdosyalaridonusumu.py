import os
import glob
import shutil
from tqdm import tqdm

base_path = r'C:\Users\alpcu\Desktop\DerinOgrenme\YOLOv8\GRAZPEDWRI-DX_dataset'
original_label_folders = ['labels/train_aug_processed', 'labels/valid_processed', 'labels/test_processed']

target_class_id = '3'   
new_class_id = '0'      


print("--- Etiket Dönüşümü Başlatılıyor ---")

for folder_name in tqdm(original_label_folders) :
    original_labels_dir = os.path.join(base_path, folder_name)
    
    new_folder_name = folder_name + '_fracture_only'
    new_labels_dir = os.path.join(base_path, new_folder_name)
    
    os.makedirs(new_labels_dir, exist_ok=True)
    
    print(f"\n[+] Klasör İşleniyor: {original_labels_dir}")
    print(f"[>] Yeni Etiketler: {new_labels_dir} adresine yazılacak.")

    processed_count = 0

    for original_filepath in glob.glob(os.path.join(original_labels_dir, "*.txt")):
        
        filename = os.path.basename(original_filepath)
        new_filepath = os.path.join(new_labels_dir, filename)
        
        with open(original_filepath, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == target_class_id:
                parts[0] = new_class_id
                
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)
        
        with open(new_filepath, 'w') as f:
            f.writelines(new_lines)
            
        processed_count += 1
    
    print(f"[>] İşlenen ve Oluşturulan Toplam Dosya Sayısı: {processed_count}")

print("\nEtiket dönüşümü başarıyla tamamlandı.")