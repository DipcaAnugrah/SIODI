"""
demo_preprocessing_psm.py — Visualisasi Belakang Layar: Preprocessing × PSM
=============================================================================
Script ini menggambarkan SECARA VISUAL bagaimana sistem memproses setiap
gambar dokumen menggunakan:
  * 4 Varian Preprocessing: CLAHE, Otsu, Adaptive, Raw Gray
  * 3 Mode PSM Tesseract: PSM 3 (Auto), PSM 6 (Uniform Block), PSM 11 (Sparse)

Output:
  - Satu grid besar per gambar (PNG) menampilkan semua kombinasi + teks OCR
  - File ringkasan teks .txt untuk setiap gambar
  - Laporan HTML interaktif yang bisa dibuka di browser

Jalankan: python demo_preprocessing_psm.py
"""

import os
import sys
import time
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent / "data gambar"
OUTPUT_DIR = Path(__file__).parent / "output_demo_preprocessing"
LANG       = "ind"
PSM_MODES  = [3, 6, 11]
PSM_LABELS = {
    3:  "PSM 3 — Auto\n(Fully automatic page\nsegmentation)",
    6:  "PSM 6 — Uniform Block\n(Anggap 1 blok teks\nseragam)",
    11: "PSM 11 — Sparse Text\n(Cari teks di mana\nsaja)",
}

# ---------------------------------------------------------------------------
# PREPROCESSING — sama persis dengan sistem utama (preprocessing.py)
# ---------------------------------------------------------------------------

def build_variants(gray):
    """Hasilkan 4 varian preprocessing dari gambar grayscale."""
    # A. CLAHE
    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    clahe_img = clahe_obj.apply(gray)

    # B. Otsu Global Thresholding
    _, otsu_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C. Adaptive Gaussian Thresholding
    denoised     = cv2.fastNlMeansDenoising(gray, h=10)
    adaptive_img = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )

    # D. Raw Grayscale (baseline)
    raw_gray = gray.copy()

    return [
        ("clahe",    "CLAHE",    clahe_img),
        ("otsu",     "Otsu",     otsu_img),
        ("adaptive", "Adaptive", adaptive_img),
        ("raw_gray", "Raw Gray", raw_gray),
    ]


def load_image(image_path):
    """Baca gambar dengan EXIF correction."""
    try:
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"  [ERROR] Gagal membaca gambar: {e}")
        return None


def run_ocr(img_gray, psm):
    """Jalankan Tesseract OCR."""
    config = f"--oem 3 --psm {psm} -l {LANG}"
    try:
        text  = pytesseract.image_to_string(img_gray, config=config)
        score = sum(1 for c in text if c.isalnum())
        return text.strip(), score
    except Exception as e:
        return f"[OCR Error: {e}]", 0


# ---------------------------------------------------------------------------
# VISUALISASI — Grid PNG
# ---------------------------------------------------------------------------
CELL_W     = 560
CELL_H     = 520
IMG_H      = 220
PADDING    = 18
HEADER_H   = 160
TOP_BANNER = 110
LEFT_LABEL = 200
TOP_LABEL  = 90

BG_DARK  = (18,  20,  40)
BG_CARD  = (28,  32,  58)
BG_HDR   = (12,  14,  30)
ACCENT   = (99, 179, 237)
GOLD     = (245, 197,  66)
GREEN    = ( 72, 199, 142)
RED_L    = (245, 101, 101)
WHITE    = (255, 255, 255)
GRAY     = (160, 170, 190)
T_DARK   = ( 18,  20,  40)


def get_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def cv2_to_pil(img):
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def draw_rr(draw, xy, radius, fill, outline=None, width=2):
    draw.rounded_rectangle(list(xy), radius=radius, fill=fill,
                            outline=outline, width=width)


def build_grid_image(image_name, orig_img, variants, ocr_results):
    n_v = 4
    n_p = 3
    total_w = LEFT_LABEL + n_v * CELL_W + PADDING * 2
    total_h = TOP_BANNER + HEADER_H + TOP_LABEL + n_p * CELL_H + PADDING * 2

    canvas = Image.new("RGB", (total_w, total_h), color=BG_DARK)
    draw   = ImageDraw.Draw(canvas)

    f_title = get_font(28)
    f_sub   = get_font(16)
    f_label = get_font(15)
    f_small = get_font(13)
    f_tiny  = get_font(11)
    f_score = get_font(14)
    f_mono  = get_font(12)
    f_big   = get_font(36)

    # ── TOP BANNER ──────────────────────────────────────────────────────────
    draw.rectangle([0, 0, total_w, TOP_BANNER], fill=BG_HDR)
    draw.rectangle([0, 0, total_w, 4], fill=ACCENT)
    draw.text((PADDING + 10, 18),
              "Visualisasi Preprocessing x PSM — Belakang Layar Sistem OCR",
              font=f_title, fill=WHITE)
    draw.text((PADDING + 10, 58),
              f"File: {image_name}   |   4 Varian Preprocessing  x  3 Mode PSM  =  12 Kombinasi OCR",
              font=f_sub, fill=GRAY)
    draw.text((PADDING + 10, 82),
              f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}   |   Tesseract OEM 3, Bahasa: ind",
              font=f_tiny, fill=(100, 115, 140))

    badge_x = total_w - 340
    draw_rr(draw, [badge_x, 14, total_w - 16, TOP_BANNER - 14],
            radius=10, fill=(30, 38, 70), outline=ACCENT)
    draw.text((badge_x + 14, 22), "KOMBINASI:", font=f_tiny, fill=GRAY)
    draw.text((badge_x + 14, 38), "12", font=f_big, fill=GOLD)
    draw.text((badge_x + 14, 78), "4 Variasi x 3 PSM", font=f_small, fill=GRAY)

    # ── HEADER: gambar original + alur ──────────────────────────────────────
    header_y = TOP_BANNER
    draw.rectangle([0, header_y, total_w, header_y + HEADER_H], fill=BG_CARD)
    draw.rectangle([0, header_y, total_w, header_y + 2], fill=(40, 50, 80))

    prev_w = 220
    h, w   = orig_img.shape[:2]
    scale  = min(prev_w / w, (HEADER_H - 20) / h)
    pw, ph = int(w * scale), int(h * scale)
    orig_pil = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    orig_pil = orig_pil.resize((pw, ph), Image.LANCZOS)
    px = PADDING + LEFT_LABEL // 2 - pw // 2
    py = header_y + (HEADER_H - ph) // 2
    canvas.paste(orig_pil, (px, py))
    draw.rectangle([px-2, py-2, px+pw+2, py+ph+2], outline=ACCENT, width=2)
    draw.text((px, py - 18), "Original", font=f_tiny, fill=ACCENT)

    info_x = PADDING + LEFT_LABEL + 20
    info_y = header_y + 16
    draw.text((info_x, info_y), "Alur Proses Sistem:", font=f_label, fill=WHITE)
    steps = [
        ("1", "Baca gambar + EXIF transpose (koreksi rotasi kamera HP)"),
        ("2", "Koreksi orientasi via OCR scoring (0/90/180/270 derajat)"),
        ("3", "Deskewing — koreksi kemiringan via Hough Line Transform"),
        ("4", "Konversi ke Grayscale"),
        ("5", "Hasilkan 4 varian preprocessing"),
        ("6", "OCR tiap varian x 3 PSM = 12 kombinasi, pilih skor terbaik"),
    ]
    for i, (num, step) in enumerate(steps):
        sy = info_y + 28 + i * 20
        draw_rr(draw, [info_x, sy, info_x+18, sy+16], radius=4, fill=ACCENT)
        draw.text((info_x+4, sy+1), num, font=get_font(11), fill=T_DARK)
        draw.text((info_x+24, sy), step, font=f_tiny, fill=GRAY)

    # ── LABEL VARIAN (baris atas) ───────────────────────────────────────────
    label_y = TOP_BANNER + HEADER_H
    draw.rectangle([0, label_y, total_w, label_y + TOP_LABEL], fill=(20, 24, 45))
    draw.rectangle([0, label_y, total_w, label_y + 2], fill=(40, 50, 80))

    var_defs = [
        ("CLAHE",    "Contrast Limited\nAdaptive Hist. Eq.",    (99, 179, 237)),
        ("Otsu",     "Binarisasi Global\n(Otsu Threshold)",     (245, 197, 66)),
        ("Adaptive", "Binarisasi Lokal\n(Adaptive Gaussian)",  (72, 199, 142)),
        ("Raw Gray", "Grayscale Asli\n(Tanpa threshold)",       (196, 130, 245)),
    ]
    for vi, (name, desc, color) in enumerate(var_defs):
        draw.rectangle(
            [LEFT_LABEL + vi*CELL_W + 4, label_y,
             LEFT_LABEL + (vi+1)*CELL_W - 4, label_y + 4],
            fill=color)
        cx = LEFT_LABEL + vi*CELL_W + 20
        draw.ellipse([cx, label_y+10, cx+30, label_y+40], fill=color)
        draw.text((cx+9, label_y+13), str(vi+1), font=f_label, fill=T_DARK)
        draw.text((LEFT_LABEL + vi*CELL_W + 55, label_y+14), name,
                  font=f_label, fill=color)
        for li, ln in enumerate(desc.split("\n")):
            draw.text((LEFT_LABEL + vi*CELL_W + 55, label_y+36+li*16),
                      ln, font=f_tiny, fill=GRAY)

    # ── LABEL PSM (kolom kiri) ──────────────────────────────────────────────
    grid_top   = label_y + TOP_LABEL
    psm_colors = [(99, 179, 237), (245, 197, 66), (72, 199, 142)]
    for pi, psm in enumerate(PSM_MODES):
        ry = grid_top + pi * CELL_H
        draw.rectangle([0, ry, LEFT_LABEL, ry+CELL_H], fill=BG_HDR)
        draw.rectangle([0, ry, 4, ry+CELL_H], fill=psm_colors[pi])
        cy = ry + CELL_H // 2
        badge_y = cy - 40
        draw_rr(draw, [12, badge_y, LEFT_LABEL-10, badge_y+34],
                radius=8, fill=psm_colors[pi])
        draw.text((LEFT_LABEL//2 - 20, badge_y+7),
                  f"PSM {psm}", font=f_label, fill=T_DARK)
        for li, ln in enumerate(PSM_LABELS[psm].split("\n")):
            draw.text((12, badge_y+42+li*16), ln, font=f_tiny, fill=GRAY)

    # ── SEL GRID ────────────────────────────────────────────────────────────
    var_colors = [
        (99, 179, 237), (245, 197, 66), (72, 199, 142), (196, 130, 245)
    ]
    for pi, psm in enumerate(PSM_MODES):
        for vi, (var_key, var_name, var_img) in enumerate(variants):
            text, score = ocr_results.get((var_key, psm), ("", 0))
            x0 = LEFT_LABEL + vi * CELL_W
            y0 = grid_top   + pi * CELL_H
            x1 = x0 + CELL_W
            y1 = y0 + CELL_H
            vcolor = var_colors[vi]

            draw.rectangle([x0+2, y0+2, x1-2, y1-2], fill=BG_CARD)
            border_c = vcolor if score > 100 else (50, 60, 90)
            border_w = 2 if score > 100 else 1
            draw.rectangle([x0+2, y0+2, x1-2, y1-2],
                           outline=border_c, width=border_w)

            # Preview gambar
            prev_h = IMG_H
            prev_w_cell = CELL_W - 2 * PADDING
            img_pil = cv2_to_pil(var_img)
            img_pil = img_pil.resize((prev_w_cell, prev_h), Image.LANCZOS)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            canvas.paste(img_pil, (x0+PADDING, y0+PADDING))
            draw.rectangle(
                [x0+PADDING-1, y0+PADDING-1,
                 x0+PADDING+prev_w_cell+1, y0+PADDING+prev_h+1],
                outline=(60, 75, 110), width=1)

            # Skor
            score_y   = y0 + PADDING + prev_h + 6
            score_col = GREEN if score > 150 else (GOLD if score > 60 else RED_L)
            draw_rr(draw, [x0+PADDING, score_y, x0+PADDING+120, score_y+22],
                    radius=5, fill=(30, 38, 70), outline=score_col)
            draw.text((x0+PADDING+6, score_y+3),
                      f"Skor: {score}", font=f_score, fill=score_col)
            draw.text((x0+PADDING+128, score_y+4),
                      f"{var_name} x PSM {psm}", font=f_tiny, fill=(100, 115, 140))

            # Teks OCR (preview)
            text_y = score_y + 28
            lines  = [l for l in (text.split("\n") if text else []) if l.strip()]
            if not lines:
                lines = ["(tidak ada teks terdeteksi)"]
            draw.rectangle([x0+PADDING, text_y, x1-PADDING, y1-PADDING],
                           fill=(22, 26, 48), outline=(40, 50, 80))
            text_area_h = y1 - text_y - PADDING
            max_lines   = (text_area_h - 8) // 14
            for li, line in enumerate(lines[:max_lines]):
                ly = text_y + 5 + li * 14
                if len(line) > 50:
                    line = line[:47] + "..."
                draw.text((x0+PADDING+5, ly), line, font=f_mono,
                          fill=(200, 210, 230))
            if len(lines) > max_lines:
                draw.text((x0+PADDING+5, y1-PADDING-16),
                          f"  ... ({len(lines)-max_lines} baris lagi)",
                          font=f_tiny, fill=(80, 95, 120))

    # Grid lines
    sep = (40, 50, 80)
    for vi in range(1, 4):
        lx = LEFT_LABEL + vi * CELL_W
        draw.rectangle([lx, grid_top, lx+1, grid_top+n_p*CELL_H], fill=sep)
    for pi in range(1, 3):
        ly = grid_top + pi * CELL_H
        draw.rectangle([LEFT_LABEL, ly, total_w, ly+1], fill=sep)
    draw.rectangle([LEFT_LABEL-1, grid_top, LEFT_LABEL+1,
                    grid_top+n_p*CELL_H], fill=ACCENT)
    draw.rectangle([0, total_h-4, total_w, total_h], fill=ACCENT)

    return canvas


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def build_html_report(results_all, output_dir):
    html_path = output_dir / "laporan_preprocessing_psm.html"

    psm_badge = {
        3:  '<span style="background:#3b82f6;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">PSM 3 Auto</span>',
        6:  '<span style="background:#f59e0b;color:#000;padding:2px 8px;border-radius:4px;font-size:12px">PSM 6 Uniform</span>',
        11: '<span style="background:#10b981;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">PSM 11 Sparse</span>',
    }
    var_badge = {
        "clahe":    '<span style="background:#3b82f6;color:#fff;padding:2px 6px;border-radius:4px;font-size:12px">CLAHE</span>',
        "otsu":     '<span style="background:#f59e0b;color:#000;padding:2px 6px;border-radius:4px;font-size:12px">Otsu</span>',
        "adaptive": '<span style="background:#10b981;color:#fff;padding:2px 6px;border-radius:4px;font-size:12px">Adaptive</span>',
        "raw_gray": '<span style="background:#8b5cf6;color:#fff;padding:2px 6px;border-radius:4px;font-size:12px">Raw Gray</span>',
    }

    cards_html = ""
    for item in results_all:
        img_name  = item["image_name"]
        grid_name = item["grid_path"].name
        best_score = max((s for _, _, s, _ in item["combos"]), default=0)
        combo_rows = ""
        for var_key, psm, score, text in sorted(item["combos"], key=lambda x: -x[2]):
            is_best  = (score == best_score and best_score > 0)
            hl       = "border-left:4px solid #22d3ee;" if is_best else ""
            best_tag = "<span style='color:#22d3ee;font-size:11px'> * TERBAIK</span>" if is_best else ""
            tp = (text[:300] + "...") if len(text) > 300 else text
            tp = tp.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            combo_rows += f"""
            <tr style="background:#1e2240;{hl}">
              <td style="padding:8px 12px">{var_badge.get(var_key,'')}</td>
              <td style="padding:8px 12px">{psm_badge.get(psm,'')}</td>
              <td style="padding:8px 12px;color:#{'22d3ee' if is_best else 'a0aec0'};font-weight:{'bold' if is_best else 'normal'}">{score}{best_tag}</td>
              <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:#cbd5e0;white-space:pre-wrap;max-width:500px">{tp}</td>
            </tr>"""

        cards_html += f"""
        <div class="card">
          <div class="card-header">
            <h2>{img_name}</h2>
            <span class="badge">12 Kombinasi</span>
          </div>
          <div style="margin-bottom:16px">
            <a href="{grid_name}" target="_blank">
              <img src="{grid_name}" style="max-width:100%;border-radius:8px;border:1px solid #334">
            </a>
            <p style="color:#718096;font-size:12px;margin-top:4px">Klik untuk zoom. Grid 4x3: semua kombinasi preprocessing x PSM</p>
          </div>
          <h3 style="color:#90cdf4;margin-bottom:8px">Tabel Hasil OCR per Kombinasi (diurutkan skor tertinggi)</h3>
          <div style="overflow-x:auto">
          <table>
            <thead><tr><th>Preprocessing</th><th>Mode PSM</th><th>Skor</th><th>Teks OCR</th></tr></thead>
            <tbody>{combo_rows}</tbody>
          </table>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Laporan Preprocessing x PSM — Sistem OCR Dokumen</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:#0f1123;color:#e2e8f0;line-height:1.6;padding:0 0 60px 0}}
.hero{{background:linear-gradient(135deg,#0f1123 0%,#1a1f45 50%,#0f2944 100%);border-bottom:1px solid #2d3748;padding:56px 60px 40px;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899)}}
.hero h1{{font-size:2rem;font-weight:700;margin-bottom:8px;background:linear-gradient(90deg,#63b3ed,#9f7aea);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero p{{color:#a0aec0;font-size:1rem;max-width:700px}}
.chips{{margin-top:20px;display:flex;gap:10px;flex-wrap:wrap}}
.chip{{background:#1e2240;border:1px solid #334;padding:6px 16px;border-radius:999px;font-size:13px;color:#90cdf4}}
.section{{max-width:1400px;margin:40px auto 0;padding:0 40px}}
.explain-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:40px}}
.explain-card{{background:#151829;border:1px solid #2d3748;border-radius:12px;padding:20px;border-top:3px solid}}
.explain-card h3{{font-size:14px;font-weight:700;margin-bottom:8px}}
.explain-card p{{font-size:13px;color:#a0aec0}}
.explain-card code{{background:#1e2240;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:12px}}
.card{{background:#151829;border:1px solid #2d3748;border-radius:16px;padding:28px;margin-bottom:40px}}
.card-header{{display:flex;align-items:center;gap:16px;margin-bottom:20px}}
.card-header h2{{font-size:1.25rem;font-weight:700;color:#f7fafc}}
.badge{{background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600}}
h3{{font-size:1rem;font-weight:600}}
table{{width:100%;border-collapse:collapse;background:#0f1123;border-radius:8px;overflow:hidden}}
th{{background:#1a1f45;color:#90cdf4;font-size:13px;font-weight:600;padding:12px 14px;text-align:left;border-bottom:1px solid #2d3748}}
td{{border-bottom:1px solid #1a2035;vertical-align:top}}
tr:last-child td{{border-bottom:none}}
tr:hover{{background:#1a2035 !important}}
.footer{{text-align:center;color:#4a5568;font-size:12px;margin-top:60px;padding:20px;border-top:1px solid #1a2035}}
</style>
</head>
<body>
<div class="hero">
  <h1>Laporan Preprocessing x PSM — Belakang Layar Sistem OCR</h1>
  <p>Dokumentasi visual bagaimana sistem memproses setiap gambar dokumen (KTP/SIM/KK) menggunakan
  4 varian preprocessing citra dikombinasikan dengan 3 mode PSM Tesseract, menghasilkan
  <strong>12 kombinasi OCR per gambar</strong>.</p>
  <div class="chips">
    <div class="chip">{len(results_all)} Gambar Diproses</div>
    <div class="chip">4 Preprocessing Variants</div>
    <div class="chip">3 PSM Modes (3, 6, 11)</div>
    <div class="chip">12 Kombinasi / Gambar</div>
    <div class="chip">Bahasa: ind (Tesseract)</div>
  </div>
</div>

<div class="section">
  <h2 style="margin-bottom:20px;color:#90cdf4">4 Varian Preprocessing</h2>
  <div class="explain-grid">
    <div class="explain-card" style="border-top-color:#3b82f6">
      <h3 style="color:#63b3ed">CLAHE</h3>
      <p><strong>Contrast Limited Adaptive Histogram Equalization</strong><br>
      Meningkatkan kontras lokal pada setiap tile (16x16 px). Efektif untuk pencahayaan tidak merata.
      Parameter: <code>clipLimit=2.5</code>, <code>tileGridSize=(16,16)</code>.</p>
    </div>
    <div class="explain-card" style="border-top-color:#f59e0b">
      <h3 style="color:#fbbf24">Otsu Thresholding</h3>
      <p><strong>Binarisasi Global Otsu</strong><br>
      Memilih threshold binarisasi optimal dari histogram intensitas global.
      Menghasilkan gambar hitam-putih. Kode: <code>THRESH_BINARY + THRESH_OTSU</code>.</p>
    </div>
    <div class="explain-card" style="border-top-color:#10b981">
      <h3 style="color:#34d399">Adaptive Gaussian</h3>
      <p><strong>Binarisasi Lokal Adaptif</strong><br>
      Tiap area kecil punya threshold sendiri. Didahului denoising
      (<code>fastNlMeansDenoising</code>). Efektif untuk pencahayaan tidak merata.</p>
    </div>
    <div class="explain-card" style="border-top-color:#8b5cf6">
      <h3 style="color:#a78bfa">Raw Grayscale</h3>
      <p><strong>Baseline tanpa enhancement</strong><br>
      Konversi BGR ke grayscale langsung. Digunakan sebagai baseline perbandingan.
      Kode: <code>cv2.COLOR_BGR2GRAY</code>.</p>
    </div>
  </div>

  <h2 style="margin-bottom:20px;color:#90cdf4">3 Mode PSM Tesseract</h2>
  <div class="explain-grid">
    <div class="explain-card" style="border-top-color:#3b82f6">
      <h3 style="color:#63b3ed">PSM 3 — Fully Automatic</h3>
      <p>Tesseract melakukan segmentasi halaman penuh dan otomatis (OSD + layout detection).
      Baik sebagai fallback umum dan untuk KK berbentuk tabel multi-kolom.</p>
    </div>
    <div class="explain-card" style="border-top-color:#f59e0b">
      <h3 style="color:#fbbf24">PSM 6 — Uniform Text Block</h3>
      <p>Tesseract mengasumsikan gambar adalah satu blok teks seragam.
      Optimal untuk KTP/SIM yang memiliki field terstruktur dan rapi.</p>
    </div>
    <div class="explain-card" style="border-top-color:#10b981">
      <h3 style="color:#34d399">PSM 11 — Sparse Text</h3>
      <p>Tesseract mencari teks di mana saja tanpa asumsi layout.
      Efektif untuk dokumen dengan teks tersebar, tidak dalam blok rapat.</p>
    </div>
  </div>

  <h2 style="margin-bottom:24px;color:#90cdf4">Hasil per Gambar</h2>
  {cards_html}
</div>

<div class="footer">
  Laporan dibuat oleh <strong>demo_preprocessing_psm.py</strong> —
  Sistem OCR Dokumen v6.0 — {datetime.now().strftime('%d %B %Y, %H:%M')}
</div>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def process_image(image_path, output_dir):
    print(f"\n{'='*65}")
    print(f"  Memproses: {image_path.name}")
    print(f"{'='*65}")

    img = load_image(str(image_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("  [1/3] Membuat 4 varian preprocessing...")
    variants = build_variants(gray)

    print("  [2/3] Menjalankan 12 kombinasi OCR (4 varian x 3 PSM)...")
    ocr_results = {}
    combos      = []
    best_score_overall = 0
    best_combo         = None

    for var_key, var_name, var_img in variants:
        for psm in PSM_MODES:
            t0 = time.time()
            text, score = run_ocr(var_img, psm)
            elapsed = time.time() - t0
            ocr_results[(var_key, psm)] = (text, score)
            combos.append((var_key, psm, score, text))
            print(f"    [{var_name:10s} x PSM {psm:2d}]  skor={score:4d}  ({elapsed:.2f}s)")
            if score > best_score_overall:
                best_score_overall = score
                best_combo = (var_key, var_name, psm, text, score)

    print("  [3/3] Membuat grid visualisasi...")
    grid_img  = build_grid_image(image_path.name, img, variants, ocr_results)
    grid_path = output_dir / f"grid_{image_path.stem}.png"
    grid_img.save(str(grid_path), "PNG")
    print(f"        Tersimpan: {grid_path.name}")

    txt_path = output_dir / f"ocr_best_{image_path.stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"HASIL OCR TERBAIK — {image_path.name}\n{'='*60}\n\n")
        if best_combo:
            f.write(f"Kombinasi Terbaik: {best_combo[1]} x PSM {best_combo[2]}\n")
            f.write(f"Skor             : {best_combo[4]}\n\n{'—'*60}\n")
            f.write(best_combo[3])
        f.write(f"\n\n{'='*60}\nSEMUA KOMBINASI (diurutkan skor tertinggi):\n\n")
        for var_key, psm, score, text in sorted(combos, key=lambda x: -x[2]):
            f.write(f"[{var_key} x PSM {psm}] skor={score}\n")
            f.write(text[:200] + ("\n... (dipotong)\n" if len(text) > 200 else "\n"))
            f.write("—"*40 + "\n")

    if best_combo:
        print(f"\n  * TERBAIK: [{best_combo[1]} x PSM {best_combo[2]}]  skor={best_combo[4]}")
        print(f"  Teks awal: {best_combo[3][:80]!r}")

    return {
        "image_name": image_path.name,
        "grid_path":  grid_path,
        "txt_path":   txt_path,
        "combos":     combos,
        "best":       best_combo,
    }


def main():
    print("\n" + "+"*65)
    print("  DEMO: Belakang Layar — Preprocessing x PSM")
    print("  Sistem OCR Dokumen (KTP/SIM/KK)")
    print("+"*65)

    if not DATA_DIR.exists():
        print(f"\n[ERROR] Folder tidak ditemukan: {DATA_DIR}")
        sys.exit(1)

    images = sorted([
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ])
    if not images:
        print(f"[ERROR] Tidak ada gambar di: {DATA_DIR}")
        sys.exit(1)

    print(f"\nDitemukan {len(images)} gambar:")
    for img in images:
        print(f"  - {img.name}  ({img.stat().st_size//1024} KB)")

    try:
        ver = pytesseract.get_tesseract_version()
        print(f"\nTesseract versi: {ver}")
    except Exception as e:
        print(f"\n[WARNING] Tesseract: {e}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")

    all_results = []
    t_start = time.time()
    for img_path in images:
        result = process_image(img_path, OUTPUT_DIR)
        if result:
            all_results.append(result)

    print(f"\n{'—'*65}")
    print("  Membuat laporan HTML interaktif...")
    html_path = build_html_report(all_results, OUTPUT_DIR)
    print(f"  Tersimpan: {html_path}")

    t_total = time.time() - t_start
    print(f"\n{'+'*65}")
    print(f"  SELESAI  |  {len(all_results)} gambar  |  {len(all_results)*12} kombinasi OCR")
    print(f"  Total waktu: {t_total:.1f} detik")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Laporan: {html_path.name}")
    print(f"{'+'*65}\n")

    import webbrowser
    try:
        webbrowser.open(str(html_path))
        print("  Laporan dibuka di browser.")
    except Exception:
        print(f"  Buka manual: {html_path}")


if __name__ == "__main__":
    main()
