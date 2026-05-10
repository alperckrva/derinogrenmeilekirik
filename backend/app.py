"""
============================================================
  RÖNTGEN KIRIK ANALİZ — FLASK BACKEND
============================================================
  Çalıştırmak için:  python app.py
  Adres            :  http://localhost:5000
============================================================
"""

import sys
import os

# pipeline.py'nın bulunduğu üst klasörü Python path'e ekle
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(BASE_DIR))

from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import torch
import datetime

from pipeline import (
    load_kiriktespit, load_anatomik, load_kiriktur, load_yolo,
    tahmin_kiriktespit, tahmin_anatomik, tahmin_kiriktur,
    hesapla_aciliyet,
)

# ============================================================
# UYGULAMA AYARLARI
# ============================================================

app = Flask(__name__)

BEST_DIR   = os.path.join(BASE_DIR, "best")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# MODELLERİ UYGULAMA BAŞLARKEN BİR KEZ YÜKLE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Cihaz: {device}")
print("📦 Modeller yükleniyor...\n")

m_tespit   = load_kiriktespit(os.path.join(BEST_DIR, "kiriktespit.pth"), device)
m_anatomik = load_anatomik(os.path.join(BEST_DIR, "anatomik.pth"), device)
m_kiriktur = load_kiriktur(os.path.join(BEST_DIR, "kiriktur.pth"), device)
m_yolo     = load_yolo(os.path.join(BEST_DIR, "kiriksayisi.pt"))

print("\n✅ Tüm modeller yüklendi! Sunucu hazır.\n")


# ============================================================
# YOLO BOUNDING BOX ÇİZİMİ
# ============================================================

def draw_yolo_boxes(image_path: str):
    """
    YOLO tahminlerini çalıştırır, tespit edilen kırıkları
    görüntü üzerine çizer ve (base64_img, kirik_sayisi) döner.
    """
    results = m_yolo.predict(source=image_path, conf=0.25, verbose=False)
    img     = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    boxes = results[0].boxes
    count = 0

    if boxes is not None and len(boxes) > 0:
        count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf_pct = box.conf[0].item() * 100

            # Yarı saydam dolgu
            draw.rectangle([x1, y1, x2, y2], fill=(255, 50, 50, 40))
            # Dış çerçeve
            draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60, 230), width=3)

            # Etiket arka planı
            label = f" Kırık %{conf_pct:.0f} "
            label_y = max(0, y1 - 24)
            draw.rectangle([x1, label_y, x1 + len(label) * 8, label_y + 22],
                           fill=(255, 60, 60, 200))
            draw.text((x1 + 2, label_y + 3), label, fill=(255, 255, 255, 255))

    # Overlay'i birleştir
    combined = Image.alpha_composite(img, overlay).convert("RGB")

    buffer = io.BytesIO()
    combined.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_b64}", max(count, 1)


def image_to_base64(image_path: str) -> str:
    """Orijinal görüntüyü base64'e çevirir (kırık yok durumu)."""
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ============================================================
# ROTALAR
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Görüntü bulunamadı"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    # Yüklenen dosyayı kaydet
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = f"upload_{timestamp}_{file.filename}"
    filepath   = os.path.join(UPLOAD_DIR, safe_name)
    file.save(filepath)

    try:
        tarih = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # ---- ADIM 1: Kırık var mı? ----
        tespit, tespit_conf = tahmin_kiriktespit(m_tespit, filepath, device)

        if tespit == "Kirik Yok":
            return jsonify({
                "kirik_var"       : False,
                "annotated_image" : image_to_base64(filepath),
                "goruntu"         : file.filename,
                "tarih"           : tarih,
                "tespit_conf"     : round(tespit_conf, 1),
            })

        # ---- ADIM 2: YOLO — kırık yerleri işaretle + say ----
        annotated_image, kirik_sayisi = draw_yolo_boxes(filepath)

        # ---- ADIM 3: Kırık türü ----
        kiriktur, kiriktur_conf = tahmin_kiriktur(m_kiriktur, filepath, device)

        # ---- ADIM 4: Anatomik bölge ----
        anatomik, anatomik_conf = tahmin_anatomik(m_anatomik, filepath, device)

        # ---- ADIM 5: Aciliyet skoru ----
        skor, seviye, seviye_ad, oneri, renk_emoji, stip, csayi, sbolge = \
            hesapla_aciliyet(kiriktur, anatomik, kirik_sayisi)

        genel_guven = round((tespit_conf + kiriktur_conf + anatomik_conf) / 3, 1)

        # Seviyeye göre CSS rengi
        renk_css = {1: "red", 2: "orange", 3: "yellow", 4: "green"}.get(seviye, "orange")

        return jsonify({
            "kirik_var"       : True,
            "annotated_image" : annotated_image,
            "goruntu"         : file.filename,
            "tarih"           : tarih,
            "tespit_conf"     : round(tespit_conf, 1),
            "kirik_sayisi"    : kirik_sayisi,
            "kiriktur"        : kiriktur,
            "kiriktur_conf"   : round(kiriktur_conf, 1),
            "anatomik"        : anatomik,
            "anatomik_conf"   : round(anatomik_conf, 1),
            "skor"            : skor,
            "seviye"          : seviye,
            "seviye_ad"       : seviye_ad,
            "renk"            : renk_css,
            "renk_emoji"      : renk_emoji,
            "oneri"           : oneri,
            "genel_guven"     : genel_guven,
            "formul"          : f"(Stip={stip} × Csayi={csayi}) + Sbölge={sbolge} = {skor}",
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ============================================================
# ÇALIŞTIR
# ============================================================

if __name__ == "__main__":
    print("🌐 Sunucu başlatılıyor → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
