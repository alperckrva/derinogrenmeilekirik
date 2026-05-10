# 🦴 Derin Öğrenme ile Röntgen Görüntüsünden Kırık Analizi

> Yapay zeka destekli röntgen analiz sistemi — 4 farklı derin öğrenme modeli kullanarak kırık tespiti, kırık sayımı, kırık türü sınıflandırması ve anatomik bölge tanıma gerçekleştirir. Sistem, sonunda otomatik aciliyet skoru hesaplayarak tıbbi karar destek raporu üretir.

---

## 📌 Projenin Amacı

Bu proje, acil servis ve radyoloji birimlerinde röntgen görüntülerinin hızlı ve tutarlı bir şekilde değerlendirilmesine yardımcı olmak amacıyla geliştirilmiştir. Sistem bir **karar destek aracı** olup uzman hekim değerlendirmesinin yerini almaz; ancak:

- Yoğun acil servis koşullarında **önceliklendirme (triage) sürecini** hızlandırmayı,
- Gözden kaçabilecek kırıkları **ikinci bir göz** olarak tespit etmeyi,
- Kırığın türü ve anatomik konumuna göre **aciliyet skoru** üreterek müdahale süresini optimize etmeyi

hedeflemektedir.

---

## 🧠 Kullanılan Modeller

| # | Model | Mimari | Görev | Doğruluk |
|---|-------|--------|-------|-----------|
| 1 | `kiriktespit.pth` | Custom CNN (5 Conv Block) | Kırık var mı / yok mu? | **%93** |
| 2 | `kiriksayisi.pt` | YOLOv8 (Object Detection) | Kaç kırık var? (Bounding box) | **%92** |
| 3 | `kiriktur.pth` | EfficientNet-B3 (Transfer Learning) | Kırık türü nedir? | **%55** |
| 4 | `anatomik.pth` | DenseNet121 (Transfer Learning) | Anatomik bölge hangisi? | **%99** |

### Kırık Türleri (Model 3)
- `avulsion` — Avülsiyon kırığı
- `compression_pathological` — Kompresyon / Patolojik kırık
- `dislocation_articular` — Çıkık / Eklem kırığı
- `simple_linear_oblique` — Basit / Lineer / Oblik kırık

### Anatomik Bölgeler (Model 4)
`chest` · `elbow` · `forearm` · `hand` · `humerus` · `shoulder` · `spine` · `wrist`

---

## ⚡ Aciliyet Skoru Algoritması

Sistem, 3 modelin çıktısını birleştirerek **aciliyet skoru** hesaplar:

```
Skor = (S_tip × C_sayı) + S_bölge
```

| Parametre | Açıklama |
|-----------|----------|
| `S_tip` | Kırık türünün tehlike puanı (2–5) |
| `C_sayı` | Kırık sayısı çarpanı (1.0 / 1.3 / 1.6) |
| `S_bölge` | Anatomik bölgenin risk puanı (3–5) |

| Skor | Seviye | Renk |
|------|--------|------|
| ≥ 11.0 | KRİTİK | 🔴 KIRMIZI |
| ≥ 8.5 | ÇOK ACİL | 🟠 TURUNCU |
| ≥ 6.5 | ACİL | 🟡 SARI |
| < 6.5 | STANDART | 🟢 YEŞİL |

---

## 🗂️ Proje Yapısı

```
📦 Bitirme Projesi
├── pipeline.py                     # Ana analiz pipeline'ı (4 model birleşik)
├── altincideneme.py                # Kırık var/yok CNN eğitim kodu
├── best/                           # ← Model dosyaları buraya konulmalı
│   ├── kiriktespit.pth
│   ├── anatomik.pth
│   ├── kiriktur.pth
│   └── kiriksayisi.pt
├── backend/
│   ├── app.py                      # Flask web uygulaması
│   └── templates/
│       └── index.html              # Web arayüzü
├── kirik var yok/                  # CNN eğitim grafikleri ve notlar
├── kirik tipi siniflandirma/       # EfficientNet-B3 eğitim grafikleri
├── anatomik bolge siniflandirma/   # DenseNet121 eğitim grafikleri
├── kirik yer tespiti/              # YOLO eğitim kodları ve sonuçları
└── analiz_sonuclari/               # Pipeline çıktı raporları
```

---

## 📥 Model Dosyalarını İndirme

Model dosyaları boyutları nedeniyle GitHub'da barındırılmamaktadır. Aşağıdaki Google Drive bağlantısından indirebilirsiniz:

### 🔗 [Google Drive — Model Dosyaları](https://drive.google.com/drive/folders/1LDwHErF1WhTyn7XB3BBZDgzXB3t4L7n0?usp=sharing)

İndirilen dosyaları proje kök dizininde **`best/`** adlı bir klasör oluşturup içine koyun:

| Drive'daki Dosya | Hedef Konum | Adını Değiştir |
|------------------|-------------|----------------|
| `altinci_deneme_kirik_tespit.pth` | `best/` | → `kiriktespit.pth` |
| `best_model_anatomik.pth` | `best/` | → `anatomik.pth` |
| `best_model_STAGE2_FINETUNE.pth` | `best/` | → `kiriktur.pth` |
| YOLO modeli (eğitim sonrası `.pt`) | `best/` | → `kiriksayisi.pt` |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükle

```bash
pip install torch torchvision pillow flask ultralytics
```

> GPU kullanmak isteyenler için: [PyTorch CUDA kurulum sayfası](https://pytorch.org/get-started/locally/)

### 2. Model Dosyalarını Yerleştir

Google Drive'dan indirdiğin modelleri `best/` klasörüne koy (yukarıdaki tabloya göre yeniden adlandır).

---

### ▶️ Seçenek A — Pipeline (Komut Satırı)

Birden fazla röntgen görüntüsünü toplu analiz eder, `.txt` formatında rapor üretir.

```python
# pipeline.py içindeki TEST_IMAGES listesini düzenle:
TEST_IMAGES = [
    "goruntu1.jpg",
    "goruntu2.png",
]
```

```bash
python pipeline.py
```

Raporlar `analiz_sonuclari/` klasörüne kaydedilir.

**Örnek Rapor Çıktısı:**
```
=================================================================
       RÖNTGEN KIRIK ANALİZ RAPORU
=================================================================
Görüntü      : goruntu1.jpg
Tarih/Saat   : 10/05/2026 13:45:00
-----------------------------------------------------------------
KIRIK TESPİT  : KIRIK VAR  (Model güveni: %94.3)
KIRIK SAYISI  : 1 adet
KIRIK TÜRÜ    : avulsion  (Model güveni: %72.1)
ANATOMİK BÖLGE: wrist  (Model güveni: %99.1)
-----------------------------------------------------------------
ACİLİYET SKORU    : 6.0
SEVİYE 4          : 🟢 YEŞİL -- STANDART
  Öneri           : Düşük Öncelik — Ayaktan tedavi veya rutin alçı/atel işlemleri.
=================================================================
```

---

### ▶️ Seçenek B — Web Arayüzü (Flask)

Röntgen görüntüsünü sürükle-bırak yöntemiyle yükleyebileceğin, sonuçları ve bounding box'ları görsel olarak gösteren web uygulaması.

```bash
cd backend
python app.py
```

Tarayıcıdan aç: **http://localhost:5000**

---

## 🖼️ Ekran Görüntüleri

| Web Arayüzü | Kırık Var Sonucu | Kırık Yok Sonucu |
|-------------|-----------------|-----------------|
| ![Arayüz](web%20site%20arayüzü.png) | ![Kırık Var](web%20site%20kirik%20var%20.png) | ![Kırık Yok](web%20site%20kırık%20yok%20.png) |

---

## 🛠️ Teknoloji Yığını

- **Python 3.10+**
- **PyTorch** — Model eğitimi ve çıkarım
- **Ultralytics YOLOv8** — Kırık lokalizasyonu
- **Flask** — Web backend
- **Pillow** — Görüntü işleme ve bounding box çizimi
- **torchvision** — Transfer learning modelleri (EfficientNet-B3, DenseNet121)

---

## ⚠️ Sorumluluk Reddi

Bu sistem bir **akademik bitirme projesi** ve **karar destek aracı** olarak geliştirilmiştir. Kesin tanı için mutlaka uzman hekim değerlendirmesi gereklidir. Klinik kullanım için uygun değildir.

---

## 👨‍💻 Geliştirici

**Alper Çakır**  
Bilgisayar Mühendisliği Bitirme Projesi  
2026
