"""
============================================================
  RÖNTGEN KIRIK ANALİZ PİPELINE
============================================================
  4 model kullanarak kırık analizi yapar ve aciliyet skoru hesaplar.

  Modeller:
    kiriktespit.pth   → Custom CNN        → Kırık var mı yok mu? (%93)
    kiriksayisi.pt    → YOLO              → Kaç kırık var?        (%92)
    kiriktur.pth      → EfficientNet-B3   → Kırık türü nedir?     (%55)
    anatomik.pth      → DenseNet121       → Anatomik bölge?       (%99)

  Kullanım:
    python pipeline.py
============================================================
"""

import os
import datetime
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. CUSTOM CNN MİMARİSİ (kiriktespit.pth)
# ============================================================

class KirikVarmiYokmu(nn.Module):
    def __init__(self):
        super(KirikVarmiYokmu, self).__init__()

        self.conv1 = nn.Conv2d(1, 64,  3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64,  128, 3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.bn4   = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn5   = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1     = nn.Linear(512, 1024)
        self.bn6     = nn.BatchNorm1d(1024)
        self.fc2     = nn.Linear(1024, 512)
        self.bn7     = nn.BatchNorm1d(512)
        self.fc3     = nn.Linear(512, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn6(self.fc1(x))))
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================
# 2. MODEL YÜKLEME
# ============================================================

def load_kiriktespit(path, device):
    model = KirikVarmiYokmu()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"  ✅ kiriktespit yüklendi")
    return model


def load_anatomik(path, device, num_classes=8):
    model = models.densenet121(weights=None)
    in_f  = model.classifier.in_features   # 1024
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  ✅ anatomik (DenseNet121) yüklendi")
    return model


def load_kiriktur(path, device, num_classes=4):
    model = models.efficientnet_b3(weights=None)
    in_f  = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 512),      # idx 0
        nn.BatchNorm1d(512),       # idx 1
        nn.ReLU(),                 # idx 2
        nn.Dropout(0.3),           # idx 3
        nn.Linear(512, 256),       # idx 4
        nn.BatchNorm1d(256),       # idx 5
        nn.ReLU(),                 # idx 6
        nn.Dropout(0.2),           # idx 7
        nn.Linear(256, num_classes),  # idx 8
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"  ✅ kiriktur (EfficientNet-B3) yüklendi")
    return model


def load_yolo(path):
    model = YOLO(path)
    print(f"  ✅ kiriksayisi (YOLO) yüklendi")
    return model


# ============================================================
# 3. TRANSFORM'LAR
# ============================================================

# Custom CNN → Grayscale, 416x416, normalize(0.5)
tfm_kiriktespit = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# DenseNet121 → RGB, 224x224, ImageNet norm
tfm_anatomik = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# EfficientNet-B3 → RGB, 300x300, ImageNet norm
tfm_kiriktur = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 4. SINIF ETİKETLERİ
# ============================================================

SINIF_KIRIKTESPIT = ["Kirik Yok", "Kirik Var"]

SINIF_ANATOMIK = [
    "chest", "elbow", "forearm", "hand",
    "humerus", "shoulder", "spine", "wrist"
]

SINIF_KIRIKTUR = [
    "avulsion",
    "compression_pathological",
    "dislocation_articular",
    "simple_linear_oblique"
]


# ============================================================
# 5. ACİLİYET ALGORİTMASI
# ============================================================

# Kırık Türü Puanı (Stip)
KIRIKTUR_PUAN = {
    "simple_linear_oblique":    2,
    "avulsion":                 3,
    "dislocation_articular":    4,
    "compression_pathological": 5,
}

# Anatomik Bölge Puanı (Sbölge)
ANATOMIK_PUAN = {
    "chest":    4,   # Göğüs Kafesi — Yüksek Riskli
    "elbow":    3,   # Dirsek       — Orta/Düşük
    "forearm":  3,   # Ön Kol       — Orta/Düşük
    "hand":     3,   # El           — Orta/Düşük
    "humerus":  3,   # Üst Kol      — Orta/Düşük
    "shoulder": 3,   # Omuz         — Orta/Düşük
    "spine":    5,   # Omurga       — KRİTİK
    "wrist":    3,   # El Bileği    — Orta/Düşük
}

# Kırık Sayısı Çarpanı (Csayi)
def get_csayi(n: int) -> float:
    if n == 1:   return 1.0
    elif n == 2: return 1.3
    else:        return 1.6  # 3+

def hesapla_aciliyet(kiriktur: str, anatomik: str, kirik_sayisi: int):
    stip   = KIRIKTUR_PUAN.get(kiriktur, 3)
    sbolge = ANATOMIK_PUAN.get(anatomik, 3)
    csayi  = get_csayi(kirik_sayisi)
    skor   = round((stip * csayi) + sbolge, 2)

    if skor >= 11.0:
        seviye, ad, aciklama, renk = 1, "KRİTİK", \
            "Hayati Tehlike — Resüsitasyon gerektirir. Derhal ameliyathane veya yoğun bakım hazırlığı.", \
            "🔴 KIRMIZI"
    elif skor >= 8.5:
        seviye, ad, aciklama, renk = 2, "ÇOK ACİL", \
            "Yüksek Öncelik — Maksimum 10-15 dk içinde uzman doktor müdahalesi. Ciddi stabilite kaybı.", \
            "🟠 TURUNCU"
    elif skor >= 6.5:
        seviye, ad, aciklama, renk = 3, "ACİL", \
            "Orta Öncelik — Hastanın durumu stabil ancak cerrahi sınırda. 1 saat içinde müdahale.", \
            "🟡 SARI"
    else:
        seviye, ad, aciklama, renk = 4, "STANDART", \
            "Düşük Öncelik — Hayati risk yok. Ayaktan tedavi veya rutin alçı/atel işlemleri.", \
            "🟢 YEŞİL"

    return skor, seviye, ad, aciklama, renk, stip, csayi, sbolge


# ============================================================
# 6. TAHMİN FONKSİYONLARI
# ============================================================

def tahmin_kiriktespit(model, image_path, device):
    img    = Image.open(image_path).convert('RGB')
    tensor = tfm_kiriktespit(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred  = torch.argmax(probs, dim=1).item()
        conf  = probs[0][pred].item() * 100
    return SINIF_KIRIKTESPIT[pred], conf


def tahmin_anatomik(model, image_path, device):
    img    = Image.open(image_path).convert('RGB')
    tensor = tfm_anatomik(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred  = torch.argmax(probs, dim=1).item()
        conf  = probs[0][pred].item() * 100
    return SINIF_ANATOMIK[pred], conf


def tahmin_kiriktur(model, image_path, device):
    img    = Image.open(image_path).convert('RGB')
    tensor = tfm_kiriktur(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred  = torch.argmax(probs, dim=1).item()
        conf  = probs[0][pred].item() * 100
    return SINIF_KIRIKTUR[pred], conf


def tahmin_kiriksayisi(model, image_path):
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    count   = len(results[0].boxes) if results[0].boxes is not None else 0
    return max(count, 1)  # zaten kırık var denildi, en az 1


# ============================================================
# 7. RAPOR YAZMA
# ============================================================

SEP  = "=" * 65
SEP2 = "-" * 65

def rapor_yaz(output_path: str, image_path: str, kirik_var: bool,
              tespit_conf=None, kirik_sayisi=None,
              kiriktur=None, kiriktur_conf=None,
              anatomik=None, anatomik_conf=None):

    goruntu = os.path.basename(image_path)
    tarih   = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    satirlar = [
        SEP,
        "       RÖNTGEN KIRIK ANALİZ RAPORU",
        SEP,
        f"Görüntü      : {goruntu}",
        f"Tarih/Saat   : {tarih}",
        SEP2,
    ]

    if not kirik_var:
        satirlar += [
            "SONUÇ        : KIRIK YOK",
            SEP,
        ]
    else:
        skor, seviye, ad, aciklama, renk, stip, csayi, sbolge = \
            hesapla_aciliyet(kiriktur, anatomik, kirik_sayisi)

        genel_guven = round((tespit_conf + kiriktur_conf + anatomik_conf) / 3, 1)

        satirlar += [
            f"KIRIK TESPİT  : KIRIK VAR  (Model güveni: %{tespit_conf:.1f})",
            f"KIRIK SAYISI  : {kirik_sayisi} adet",
            f"KIRIK TÜRÜ    : {kiriktur}  (Model güveni: %{kiriktur_conf:.1f})",
            f"ANATOMİK BÖLGE: {anatomik}  (Model güveni: %{anatomik_conf:.1f})",
            SEP2,
            f"ACİLİYET SKORU    : {skor}  (Genel Güven: %{genel_guven})",
            f"  Formül          : (Stip={stip} x Csayi={csayi}) + Sbölge={sbolge} = {skor}",
            f"SEVİYE {seviye}           : {renk} -- {ad}",
            f"  Öneri           : {aciklama}",
            SEP,
            "NOT: Bu sistem bir karar destek aracidir.",
            "     Kesin tani icin uzman hekim degerlendirmesi gereklidir.",
            SEP,
        ]

    rapor = "\n".join(satirlar)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rapor)

    return rapor


# ============================================================
# 8. ANA PİPELINE
# ============================================================

def run_pipeline(image_paths: list, base_dir: str = "."):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Cihaz: {device}")

    # Model yolları
    KIRIKTESPIT_PATH = os.path.join(base_dir, "best", "kiriktespit.pth")
    ANATOMIK_PATH    = os.path.join(base_dir, "best", "anatomik.pth")
    KIRIKTUR_PATH    = os.path.join(base_dir, "best", "kiriktur.pth")
    KIRIKSAYISI_PATH = os.path.join(base_dir, "best", "kiriksayisi.pt")

    # Modelleri yükle
    print("\n📦 Modeller yükleniyor...")
    m_tespit   = load_kiriktespit(KIRIKTESPIT_PATH, device)
    m_anatomik = load_anatomik(ANATOMIK_PATH, device)
    m_kiriktur = load_kiriktur(KIRIKTUR_PATH, device)
    m_yolo     = load_yolo(KIRIKSAYISI_PATH)
    print("✅ Tüm modeller hazır!\n")

    # Çıktı klasörü
    output_dir = os.path.join(base_dir, "analiz_sonuclari")
    os.makedirs(output_dir, exist_ok=True)

    print(SEP)
    print(f"{'ANALİZ BAŞLIYOR':^65}")
    print(SEP)

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"\n⚠️  Görüntü bulunamadı: {image_path}")
            continue

        goruntu_adi = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{goruntu_adi}_rapor.txt")

        print(f"\n📸 İşleniyor: {os.path.basename(image_path)}")

        # ADIM 1 — Kırık var mı yok mu?
        tespit, tespit_conf = tahmin_kiriktespit(m_tespit, image_path, device)
        print(f"  1️⃣  Kırık Tespiti  : {tespit} (%{tespit_conf:.1f})")

        if tespit == "Kirik Yok":
            print(f"      ✅ KIRIK YOK → sonraki adımlar atlandı")
            rapor = rapor_yaz(output_path, image_path, kirik_var=False,
                              tespit_conf=tespit_conf)
        else:
            # ADIM 2 — Kaç kırık var? (YOLO)
            kirik_sayisi = tahmin_kiriksayisi(m_yolo, image_path)
            print(f"  2️⃣  Kırık Sayısı   : {kirik_sayisi} adet")

            # ADIM 3 — Kırık türü
            kiriktur, kiriktur_conf = tahmin_kiriktur(m_kiriktur, image_path, device)
            print(f"  3️⃣  Kırık Türü     : {kiriktur} (%{kiriktur_conf:.1f})")

            # ADIM 4 — Anatomik bölge
            anatomik, anatomik_conf = tahmin_anatomik(m_anatomik, image_path, device)
            print(f"  4️⃣  Anatomik Bölge : {anatomik} (%{anatomik_conf:.1f})")

            # Aciliyet skoru (ekrana özet)
            skor, seviye, ad, _, renk, _, _, _ = \
                hesapla_aciliyet(kiriktur, anatomik, kirik_sayisi)
            print(f"  ⚡  Aciliyet Skoru  : {skor}  →  SEVİYE {seviye}: {renk} {ad}")

            # Raporu yaz
            rapor = rapor_yaz(
                output_path, image_path, kirik_var=True,
                tespit_conf=tespit_conf,
                kirik_sayisi=kirik_sayisi,
                kiriktur=kiriktur, kiriktur_conf=kiriktur_conf,
                anatomik=anatomik, anatomik_conf=anatomik_conf,
            )

        print(f"  💾 Rapor: {output_path}")
        print(SEP2)

    print(f"\n✨ Tüm görüntüler işlendi!")
    print(f"📁 Sonuçlar: {os.path.abspath(output_dir)}\n")


# ============================================================
# 9. ÇALIŞTIR
# ============================================================

if __name__ == "__main__":

    BASE_DIR = r"C:\Users\alpcu\Desktop\Bitirme Projesi"

    TEST_IMAGES = [
        os.path.join(BASE_DIR, "kirikvarr.jpg"),
        os.path.join(BASE_DIR, "kirik.png"),
        os.path.join(BASE_DIR, "kirikyok1.jpg"),
        os.path.join(BASE_DIR, "kirikyok2.jpg"),
        os.path.join(BASE_DIR, "kirikyok3.jpg"),
        os.path.join(BASE_DIR, "kirikyok4.jpg"),
        os.path.join(BASE_DIR, "kirikyok.jpg"),
        os.path.join(BASE_DIR, "kırık1_avulsion.png"),
        os.path.join(BASE_DIR, "kırık2_avulsion.png"),
        os.path.join(BASE_DIR, "kırık3_pathological.png"),
        os.path.join(BASE_DIR, "kırık4_pathological.png"),
        os.path.join(BASE_DIR, "kırık5_diclocation.png"),
        os.path.join(BASE_DIR, "kırık6_diclocation.png"),
        os.path.join(BASE_DIR, "kırık7_simple.png"),
        os.path.join(BASE_DIR, "kırık8_simple.png"),
    ]

    run_pipeline(TEST_IMAGES, base_dir=BASE_DIR)
