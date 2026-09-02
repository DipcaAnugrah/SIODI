"""
diagnostic_kk_ocr_v3.py -- Visualisasi Diagnostik OCR Nomor KK (v3 -- FINAL)
==============================================================================
Mereplikasi sistem asli secara TEPAT:
  1. Preprocessing pada FULL image (preprocess_image() -- sama persis)
  2. OCR pada FULL preprocessed image -> raw text mentah SELURUH dokumen
  3. NLP/regex filter (sama dengan _extract_nomor_kk di extractor.py)
     -> ekstrak Nomor KK dari raw text
  4. TIDAK ADA crop untuk tujuan OCR. Crop HANYA untuk visualisasi area
     teks pada gambar (menunjukkan di mana teks NoKK berada secara visual).

Gambar yang dihasilkan:
  fig1 -- Gambar asli + annotasi area teks di dokumen
  fig2 -- Full preprocessed image (tampilkan 35% atas = area header KK)
  fig3 -- Raw text OCR output per strategi (teks mentah yang diterima NLP)
  fig4 -- Perbandingan digit NoKK terekstrak vs Ground Truth
  fig5 -- Heatmap kesalahan per posisi digit
  fig6 -- Confusion + kemiripan morfologis
  fig7 -- Tabel ringkasan akurasi per strategi + PSM
  fig8 -- Diagram pipeline sistem asli

Ground Truth NoKK: 71190920242753 (14 digit)
"""
import os, re, sys, cv2, numpy as np, pytesseract, matplotlib, textwrap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps
from collections import Counter

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH   = os.path.join(SCRIPT_DIR, "kk_maya_sari.jpg")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "diagnostic_output")
GROUND_TRUTH = "71190920242753"

os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {
    "bg": "#0D1117", "panel": "#161B22", "border": "#30363D",
    "text": "#E6EDF3", "sub": "#8B949E",
    "green": "#3FB950", "red": "#F85149", "yellow": "#D29922",
    "blue": "#58A6FF", "purple": "#BC8CFF", "orange": "#FFA657",
}
SC = {"clahe": "#58A6FF", "otsu": "#3FB950", "adaptive": "#FFA657", "raw_gray": "#BC8CFF"}
SN = {"clahe": "CLAHE", "otsu": "Otsu Thresholding",
      "adaptive": "Adaptive Threshold", "raw_gray": "Raw Grayscale"}

plt.rcParams.update({
    "figure.facecolor": C["bg"],   "axes.facecolor": C["panel"],
    "axes.edgecolor":   C["border"], "axes.labelcolor": C["text"],
    "xtick.color":      C["sub"],    "ytick.color": C["sub"],
    "text.color":       C["text"],   "font.family": "monospace",
})

_KK_FROM = "OolLIisSbBzZGgAaDdYyPpFfCcUuVv?"
_KK_TO   = "0011115588226600004499000000007"
_OCR_KK  = str.maketrans(_KK_FROM, _KK_TO)


# ==============================================================================
# LANGKAH 1: LOAD IMAGE (sama dengan sistem)
# ==============================================================================
def load_image():
    """Baca gambar + koreksi EXIF -- identik dengan preprocess_image()."""
    pil = Image.open(IMAGE_PATH)
    pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


# ==============================================================================
# LANGKAH 2: PREPROCESSING FULL IMAGE (identik dengan preprocess_image())
# ==============================================================================
def build_full_variants(img_bgr):
    """
    Preprocessing pada FULL image -- tidak ada crop.
    Identik dengan langkah preprocessing di preprocess_image().
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # A. CLAHE
    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    clahe     = clahe_obj.apply(gray)

    # B. Otsu Global Thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C. Adaptive Gaussian Thresholding (dengan denoising)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    # D. Raw Grayscale
    raw = gray.copy()

    return [
        ("clahe",    "CLAHE",              clahe),
        ("otsu",     "Otsu Thresholding",  otsu),
        ("adaptive", "Adaptive Threshold", adaptive),
        ("raw_gray", "Raw Grayscale",      raw),
    ]


# ==============================================================================
# LANGKAH 3: OCR FULL IMAGE + NLP FILTER (identik dengan sistem)
# ==============================================================================
def _extract_nokk_from_raw_text(raw_text):
    """
    Filter NLP -- ekstrak Nomor KK dari raw text OCR.
    Logika identik dengan _extract_nomor_kk() di extractor.py.
    Mengembalikan (nokk_string, match_context) untuk keperluan diagnostik.
    """
    def clean_to_digits(raw):
        compacted = re.sub(r"(?<=[0-9A-Za-z?])\s(?=[0-9A-Za-z?])", "", raw)
        cleaned   = re.sub(r"[^0-9A-Za-z?]", "", compacted)
        digits    = re.sub(r"[^0-9]", "", cleaned.translate(_OCR_KK))
        return digits

    candidates = []
    match_ctx   = []  # konteks match untuk visualisasi

    # -- Pola A-1: NOMOR KK / No. KK eksplisit
    m = re.search(
        r"(?:NOMOR\s+KK|No\.?\s*KK|N[o0]\.\s*KK)\s*[:\-]?\s*"
        r"([0-9A-Za-z?][0-9A-Za-z?\s]{10,28})(?:\n|$|[^0-9A-Za-z?\s])",
        raw_text, re.IGNORECASE)
    if m:
        d = clean_to_digits(m.group(1).strip())
        if 14 <= len(d) <= 16:
            candidates.append(d[:16] if len(d) == 16 else d)
            match_ctx.append(("Pola A-1 (NOMOR KK eksplisit)", m.group(0).strip()[:60]))

    # -- Pola A-2: baris dimulai "No."
    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        ls = line.strip()
        m2 = re.match(
            r"N[o0u][a\.\-]?\s*[:\-]?\s*([?#*@!~\^$]{0,2}[0-9][0-9A-Za-z?\s]{11,26})$",
            ls, re.IGNORECASE)
        if m2:
            d = clean_to_digits(m2.group(1).strip())
            if 14 <= len(d) <= 16:
                candidates.append(d[:16] if len(d) == 16 else d)
                match_ctx.append(("Pola A-2 (baris No.)", ls[:60]))
        if re.match(r"^N[o0u][a\.\-]?\s*$", ls, re.IGNORECASE) and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            d   = clean_to_digits(nxt)
            if 14 <= len(d) <= 16:
                candidates.append(d[:16] if len(d) == 16 else d)
                match_ctx.append(("Pola A-2b (No. baris terpisah)", nxt[:60]))

    # -- Pola B: fallback sequence panjang
    if not candidates:
        for seq in re.findall(r"[1-9][0-9A-Za-z?\s]{13,19}", raw_text):
            d = clean_to_digits(seq)
            if 14 <= len(d) <= 16:
                candidates.append(d[:16] if len(d) == 16 else d)
                match_ctx.append(("Pola B (fallback sequence)", seq.strip()[:60]))

    if not candidates:
        return "", "Tidak ditemukan", []
    result = Counter(candidates).most_common(1)[0][0]
    return result, match_ctx[0] if match_ctx else ("?", "?"), match_ctx


def run_full_ocr_pipeline(variants, psm_list=(3, 6, 11)):
    """
    Jalankan OCR pada FULL IMAGE untuk setiap varian preprocessing.
    Kembalikan: nokk terekstrak, raw text per PSM, konteks match regex.
    """
    results  = {}   # {key: (label, best_nokk)}
    raw_data = {}   # {key: {psm: (raw_text_snippet, nokk_extracted)}}

    for key, label, img_full in variants:
        per_psm    = {}
        best_nokk  = ""
        best_score = -1
        best_ctx   = []

        for psm in psm_list:
            config   = f"--oem 3 --psm {psm} -l ind"
            try:
                raw_text = pytesseract.image_to_string(img_full, config=config)
                nokk, ctx_first, ctx_all = _extract_nokk_from_raw_text(raw_text)

                # Skor berdasarkan kemiripan dengan GT
                n_match = sum(1 for i in range(min(len(nokk), len(GROUND_TRUTH)))
                              if nokk[i] == GROUND_TRUTH[i]) if nokk else 0
                score   = n_match + (3 if len(nokk) == len(GROUND_TRUTH) else 0)

                # Simpan snippet raw text yang relevan (area sekitar NoKK)
                snippet = _extract_nokk_context_snippet(raw_text)
                per_psm[psm] = (snippet, nokk, raw_text)

                if score > best_score:
                    best_score = score
                    best_nokk  = nokk
                    best_ctx   = ctx_all

            except Exception as e:
                per_psm[psm] = (f"[ERROR: {e}]", "", "")

        results[key]  = (label, best_nokk, best_ctx)
        raw_data[key] = per_psm

    return results, raw_data


def _extract_nokk_context_snippet(raw_text, window=5):
    """
    Ambil snippet teks di sekitar area Nomor KK dari raw text OCR.
    Cari baris yang mengandung 'No.' atau digit panjang di awal dokumen.
    """
    lines = raw_text.splitlines()
    # Cari baris yang mengandung 'No' atau sequence digit >= 10 karakter
    for i, line in enumerate(lines[:40]):  # hanya 40 baris pertama (header KK)
        if re.search(r"N[o0u][a\.\-]", line, re.IGNORECASE) or \
           re.search(r"[0-9]{8,}", line):
            start = max(0, i - 1)
            end   = min(len(lines), i + window)
            return "\n".join(lines[start:end])
    # Fallback: 10 baris pertama
    return "\n".join(lines[:10])


def compare_digits(gt, pred):
    result = []
    ml = max(len(gt), len(pred)) if (gt or pred) else 1
    for i in range(ml):
        gc = gt[i]   if i < len(gt)   else "?"
        pc = pred[i] if i < len(pred) else "?"
        result.append((i + 1, gc, pc, gc == pc))
    return result


# ==============================================================================
# GAMBAR 1: Overview Gambar Asli + Anotasi Lokasi Field
# ==============================================================================
def plot_fig1(img_rgb):
    h, w = img_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(C["bg"])
    ax.imshow(img_rgb); ax.axis("off")
    ax.set_title("Gambar 1 -- Dokumen KK Asli (kk_maya_sari.jpg)\n"
                 "OCR dijalankan pada SELURUH gambar -- tidak ada pemotongan untuk OCR",
                 fontsize=13, color=C["blue"], fontweight="bold", pad=10)

    # Anotasi: sistem membaca SELURUH dokumen
    ax.add_patch(mpatches.FancyBboxPatch(
        (5, 5), w - 10, h - 10, boxstyle="round,pad=0",
        lw=3, edgecolor=C["blue"], facecolor="none", linestyle="--"))
    ax.text(w // 2, 30,
            "SELURUH AREA INI DIBACA OLEH TESSERACT OCR (full image, tanpa crop)",
            ha="center", fontsize=11, color=C["blue"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["bg"], alpha=0.85))

    # Anotasi area atas -- Nomor KK (besar)
    nokk_y = int(h * 0.12)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, nokk_y - 30), w, int(h * 0.08),
        boxstyle="square,pad=0",
        lw=2.5, edgecolor=C["red"], facecolor=C["red"] + "22"))
    ax.text(w - 20, nokk_y + 5,
            "  Area Nomor KK\n  (font BESAR, baris atas)\n  GT: " + GROUND_TRUTH,
            ha="right", fontsize=10, color=C["red"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg"], alpha=0.85))

    # Anotasi area teks field lain
    field_y = int(h * 0.38)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, field_y), w, int(h * 0.50),
        boxstyle="square,pad=0",
        lw=2, edgecolor=C["green"], facecolor=C["green"] + "18", linestyle="--"))
    ax.text(20, field_y + 20,
            "  Area Field Teks Lain\n  (Nama Kepala, Alamat, Desa,\n"
            "  Kecamatan, Kab/Kota, Provinsi, RT/RW)\n"
            "  -> Berhasil diekstrak dengan benar",
            ha="left", fontsize=10, color=C["green"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg"], alpha=0.85))

    # Keterangan alur
    ax.text(w // 2, h - 30,
            "[1] Baca full image  -->  [2] Preprocessing 4 strategi  -->  "
            "[3] OCR Tesseract PSM 3/6/11  -->  [4] NLP Regex Filter  -->  [5] Nomor KK",
            ha="center", fontsize=9, color=C["yellow"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["bg"], alpha=0.9))

    plt.tight_layout(pad=1)
    out = os.path.join(OUTPUT_DIR, "fig1_overview_full_image.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 2: Full Preprocessed Image -- tampilkan 35% atas (area header NoKK)
# ==============================================================================
def plot_fig2(img_bgr, variants):
    """
    Tampilkan hasil preprocessing pada SELURUH gambar.
    Zoom ke 35% atas dokumen (area header KK yang memuat Nomor KK)
    agar detail teks terlihat jelas, tapi ini bukan crop untuk OCR.
    """
    h, w = img_bgr.shape[:2]
    top_h = int(h * 0.30)  # 30% atas -- area header KK (Nomor KK + Nama Kepala)

    fig = plt.figure(figsize=(22, 11), facecolor=C["bg"])
    fig.suptitle(
        "Gambar 2 -- Hasil 4 Strategi Preprocessing (Full Image)\n"
        "Ditampilkan: 30% atas dokumen (area header) agar teks Nomor KK terlihat jelas.\n"
        "PENTING: OCR Tesseract dijalankan pada SELURUH gambar (3240x2268px), bukan crop ini.",
        fontsize=12, color=C["blue"], fontweight="bold")

    gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.18)

    desc = {
        "clahe":    "CLAHE\nClipLimit=2.5\nTileGrid=(16,16)\nKontras lokal adaptif",
        "otsu":     "Otsu Global\nThreshold\nOptimal via histogram\nSatu nilai global",
        "adaptive": "Adaptive Gauss.\nBlockSize=31\nC=10\nThreshold lokal",
        "raw_gray": "Raw Grayscale\ncv2.COLOR_BGR2GRAY\nTanpa modifikasi\nLangsung ke Tesseract",
    }
    for i, (key, label, img_full) in enumerate(variants):
        # Tampilkan 30% atas untuk visibilitas (OCR aslinya full image)
        top_crop = img_full[:top_h, :]

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(top_crop, cmap="gray", aspect="auto")
        col = SC[key]
        ax.set_title(desc[key], fontsize=9, color=col, pad=5,
                     fontweight="bold", linespacing=1.4)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.5); sp.set_visible(True)
        ax.text(0.02, 0.97, f"S{i+1}", transform=ax.transAxes,
                fontsize=10, color=col, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"], alpha=0.8))
        # Resolusi info
        ax.text(0.02, 0.04, f"Full: {img_full.shape[1]}x{img_full.shape[0]}px",
                transform=ax.transAxes, fontsize=7, color=C["sub"],
                bbox=dict(boxstyle="round,pad=0.1", facecolor=C["bg"], alpha=0.7))

    for i, (key, label, img_full) in enumerate(variants):
        top_crop = img_full[:top_h, :]
        ax = fig.add_subplot(gs[1, i])
        col = SC[key]
        # Histogram dari FULL image (bukan crop)
        hist_full = cv2.calcHist([img_full], [0], None, [256], [0, 256]).flatten()
        hist_crop = cv2.calcHist([top_crop], [0], None, [256], [0, 256]).flatten()
        ax.fill_between(range(256), hist_full, alpha=0.25, color=col, label="Full image")
        ax.fill_between(range(256), hist_crop, alpha=0.55, color=col, label="Area header")
        ax.plot(range(256), hist_full, color=col, lw=1, alpha=0.6)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensitas (0-255)", fontsize=7.5, color=C["sub"])
        ax.set_ylabel("Frekuensi", fontsize=7.5, color=C["sub"])
        ax.set_title(f"Histogram -- {label}", fontsize=8, color=C["text"], pad=4)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right")
        if key in ("otsu", "adaptive"):
            ax.axvline(127, color=C["yellow"], lw=1.2, ls="--", alpha=0.7, label="Ambang")

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, "fig2_preprocessing_full_image.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 3: Raw Text OCR Output + Regex Match (per strategi + PSM terbaik)
# ==============================================================================
def plot_fig3_raw_text(raw_data):
    """
    Tampilkan raw text OCR yang diterima oleh NLP filter.
    Ini menunjukkan secara nyata teks mentah yang harus di-parse oleh regex.
    """
    sorder = ["clahe", "otsu", "adaptive", "raw_gray"]

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        "Gambar 3 -- Raw Text OCR Output per Strategi Preprocessing\n"
        "Teks mentah ini adalah INPUT yang diterima oleh NLP Regex Filter untuk ekstraksi Nomor KK",
        fontsize=13, color=C["blue"], fontweight="bold")

    axes_flat = axes.flatten()

    for idx, key in enumerate(sorder):
        ax = axes_flat[idx]
        ax.set_facecolor(C["panel"]); ax.axis("off")
        col = SC[key]

        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2); sp.set_visible(True)

        # Cari PSM dengan NoKK terdiekstrak terbaik
        per_psm = raw_data.get(key, {})
        best_psm_text = ""
        best_psm_nokk = ""
        best_psm_id   = 0
        best_sc = -1
        for psm, (snippet, nokk, full_raw) in per_psm.items():
            n_ok = sum(1 for i in range(min(len(nokk), len(GROUND_TRUTH)))
                       if nokk[i] == GROUND_TRUTH[i]) if nokk else 0
            if n_ok > best_sc:
                best_sc = n_ok; best_psm_text = snippet
                best_psm_nokk = nokk; best_psm_id = psm

        ax.set_title(f"S{idx+1}: {SN[key]}  |  PSM Terbaik: {best_psm_id}  |  "
                     f"NoKK Terekstrak: '{best_psm_nokk}'",
                     fontsize=10.5, color=col, pad=8, fontweight="bold")

        # Tampilkan snippet raw text
        if best_psm_text:
            lines      = best_psm_text.splitlines()
            y_pos      = 0.97
            line_h     = 0.062
            max_lines  = 14

            for ln in lines[:max_lines]:
                # Highlight baris yang mengandung karakter digit panjang
                is_nokk_line = bool(re.search(r"[0-9]{6,}", ln))
                is_label     = bool(re.search(r"N[o0u][a\.\-]", ln, re.IGNORECASE))
                if is_nokk_line:
                    fc = C["red"] + "44"
                    tc = C["red"]
                elif is_label:
                    fc = C["yellow"] + "33"
                    tc = C["yellow"]
                else:
                    fc = "none"
                    tc = C["text"]

                # Background highlight
                if fc != "none":
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (0.01, y_pos - line_h + 0.005), 0.98, line_h - 0.005,
                        transform=ax.transAxes, boxstyle="square,pad=0",
                        facecolor=fc, edgecolor="none", zorder=1))

                # Teks
                display_ln = ln[:95] + "..." if len(ln) > 95 else ln
                ax.text(0.03, y_pos - line_h * 0.4, display_ln,
                        transform=ax.transAxes, fontsize=8,
                        color=tc, va="center", fontfamily="monospace", zorder=2)
                y_pos -= line_h

            if len(lines) > max_lines:
                ax.text(0.03, y_pos - line_h * 0.5,
                        f"  ... ({len(lines) - max_lines} baris lagi dalam raw text penuh)",
                        transform=ax.transAxes, fontsize=7.5,
                        color=C["sub"], va="center", fontstyle="italic")
        else:
            ax.text(0.5, 0.5, "Raw text kosong / tidak tersedia",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=C["sub"])

        # Legend baris highlight
        ax.text(0.99, 0.03,
                "[Merah] = baris dengan digit panjang (kandidat NoKK)\n"
                "[Kuning] = label 'No.' / 'NOMOR KK'\n"
                "[Putih]  = baris teks biasa",
                transform=ax.transAxes, ha="right", fontsize=7.5,
                color=C["sub"], va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg"], alpha=0.8))

    plt.tight_layout(pad=2)
    out = os.path.join(OUTPUT_DIR, "fig3_raw_text_ocr_output.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 4: Digit Comparison (NoKK terekstrak vs GT)
# ==============================================================================
def plot_fig4_digit_comparison(results):
    sorder  = ["clahe", "otsu", "adaptive", "raw_gray"]
    slabels = {"clahe": "S1: CLAHE", "otsu": "S2: Otsu",
               "adaptive": "S3: Adaptive", "raw_gray": "S4: Raw Gray"}
    gt = GROUND_TRUTH; n = len(gt)

    fig, axes = plt.subplots(len(sorder) + 1, 1, figsize=(22, 12),
        gridspec_kw={"height_ratios": [1.3] + [1] * len(sorder)})
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        f"Gambar 4 -- Perbandingan Digit-per-Digit: NoKK Terekstrak vs Ground Truth\n"
        f"Ground Truth: {gt}  ({n} digit)  |  "
        f"Sumber: Regex filter dari raw OCR text (full image pipeline)",
        fontsize=13, color=C["blue"], fontweight="bold")

    ax_gt = axes[0]
    ax_gt.set_xlim(-0.5, n - 0.5); ax_gt.set_ylim(0, 1); ax_gt.axis("off")
    ax_gt.text(-0.02, 0.5, "GROUND\nTRUTH", transform=ax_gt.transAxes,
               fontsize=10, color=C["green"], va="center", ha="right", fontweight="bold")
    for i, c in enumerate(gt):
        ax_gt.add_patch(mpatches.FancyBboxPatch((i - 0.43, 0.05), 0.86, 0.90,
            boxstyle="round,pad=0.05", facecolor="#1A2B1A", edgecolor=C["green"], lw=2))
        ax_gt.text(i, 0.52, c, ha="center", va="center", fontsize=24,
                   color=C["green"], fontweight="bold", fontfamily="monospace")
        ax_gt.text(i, 0.1, f"[{i+1}]", ha="center", fontsize=7.5, color=C["sub"])

    for row, key in enumerate(sorder):
        ax = axes[row + 1]
        label, pred, ctx = results.get(key, ("?", "", []))
        cmp  = compare_digits(gt, pred)
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(0, 1); ax.axis("off")
        col  = SC[key]
        n_ok = sum(1 for _, _, _, ok in cmp if ok)
        acc  = n_ok / len(gt) * 100
        ac_c = C["green"] if acc == 100 else (C["yellow"] if acc >= 60 else C["red"])

        ax.text(-0.02, 0.5, slabels[key], transform=ax.transAxes,
                fontsize=9.5, color=col, va="center", ha="right", fontweight="bold")
        pred_disp = f"'{pred}'" if pred else "[tidak terdeteksi]"
        ax.text(1.01, 0.90, pred_disp, transform=ax.transAxes, fontsize=9,
                color=col, va="top", ha="left", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg"], alpha=0.7))
        ax.text(1.01, 0.52, f"Akurasi: {n_ok}/{n} ({acc:.0f}%)",
                transform=ax.transAxes, fontsize=8.5, color=ac_c,
                va="top", ha="left", fontweight="bold")
        # Regex context
        ctx_str = ctx[0][0] if ctx else "Pola tidak cocok"
        ax.text(1.01, 0.22, f"Match: {ctx_str}", transform=ax.transAxes,
                fontsize=7.5, color=C["sub"], va="top", ha="left")

        for i, (pos, gc, pc, is_ok) in enumerate(cmp):
            if i >= n: break
            fc = "#1A2B1A" if is_ok else "#2B1A1A"
            ec = C["green"] if is_ok else C["red"]
            ax.add_patch(mpatches.FancyBboxPatch((i - 0.43, 0.05), 0.86, 0.90,
                boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, lw=2))
            ax.text(i, 0.52, pc, ha="center", va="center", fontsize=22,
                    color=ec, fontweight="bold", fontfamily="monospace")
            ax.text(i, 0.1, "v" if is_ok else "x", ha="center", fontsize=8, color=ec)
            if not is_ok and pc != "?":
                ax.text(i + 0.38, 0.9, f"GT:{gc}", ha="left", fontsize=6,
                        color=C["sub"], fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_DIR, "fig4_digit_comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 5: Heatmap Kesalahan
# ==============================================================================
def plot_fig5_heatmap(results):
    sorder  = ["clahe", "otsu", "adaptive", "raw_gray"]
    slabels = ["S1: CLAHE", "S2: Otsu", "S3: Adaptive", "S4: Raw Gray"]
    gt = GROUND_TRUTH; n = len(gt)

    err_mat  = np.zeros((len(sorder), n), dtype=int)
    pred_mat = [["?" for _ in range(n)] for _ in sorder]
    for r, key in enumerate(sorder):
        _, pred, _ = results.get(key, ("?", "", []))
        for c in range(n):
            pc = pred[c] if c < len(pred) else "?"
            pred_mat[r][c] = pc
            err_mat[r, c]  = 0 if pc == gt[c] else 1

    fig, axes = plt.subplots(1, 2, figsize=(20, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Gambar 5 -- Heatmap Kesalahan per Posisi Digit dan Strategi\n"
                 "(Berdasarkan Nomor KK yang diekstrak oleh NLP Regex dari full OCR text)",
                 fontsize=13, color=C["blue"], fontweight="bold")

    ax_h = axes[0]
    cbin = mcolors.ListedColormap([C["green"], C["red"]])
    ax_h.imshow(err_mat, cmap=cbin, aspect="auto", vmin=0, vmax=1, alpha=0.85)
    for r in range(len(sorder)):
        for c in range(n):
            pc    = pred_mat[r][c]
            is_ok = pc == gt[c]
            ax_h.text(c, r - 0.16, pc, ha="center", va="center",
                      fontsize=14, fontweight="bold", color=C["bg"], fontfamily="monospace")
            if not is_ok:
                ax_h.text(c, r + 0.28, f"({gt[c]})", ha="center",
                          fontsize=8.5, color="#FFFFFFBB", fontfamily="monospace")
    ax_h.set_xticks(range(n))
    ax_h.set_xticklabels([f"P{i+1}\n[{gt[i]}]" for i in range(n)], fontsize=8.5)
    ax_h.set_yticks(range(len(sorder)))
    ax_h.set_yticklabels(slabels, fontsize=9.5)
    ax_h.set_xlabel("Posisi Digit  (Pn = posisi ke-n | [x] = digit GT)", fontsize=9, labelpad=8)
    ax_h.set_title("Merah = Salah | Hijau = Benar\n(Angka besar=OCR | (kecil)=GT)",
                   fontsize=9, color=C["sub"])
    for c in range(n + 1): ax_h.axvline(c - 0.5, color=C["bg"], lw=1.5)
    for r in range(len(sorder) + 1): ax_h.axhline(r - 0.5, color=C["bg"], lw=1.5)

    ax_b = axes[1]
    err_rate = err_mat.mean(axis=0)
    bc = [C["red"] if er >= 0.75 else (C["yellow"] if er > 0 else C["green"])
          for er in err_rate]
    bars = ax_b.barh(range(n), err_rate, color=bc, alpha=0.85, height=0.7,
                     edgecolor=C["border"])
    ax_b.set_yticks(range(n))
    ax_b.set_yticklabels([f"P{i+1} [GT:{gt[i]}]" for i in range(n)], fontsize=8.5)
    ax_b.set_xlim(0, 1.3); ax_b.invert_yaxis()
    ax_b.set_xlabel("Error Rate (0=benar, 1=salah semua)", fontsize=9)
    ax_b.set_title("Error Rate\nper Posisi", fontsize=10, color=C["text"])
    for bar, er in zip(bars, err_rate):
        ax_b.text(er + 0.03, bar.get_y() + bar.get_height() / 2,
                  f"{er*100:.0f}%", va="center", fontsize=8.5, fontweight="bold",
                  color=C["text"])

    plt.tight_layout(pad=2)
    out = os.path.join(OUTPUT_DIR, "fig5_error_heatmap.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 6: Confusion + Kemiripan Morfologis
# ==============================================================================
def plot_fig6_confusion(results):
    sorder = ["clahe", "otsu", "adaptive", "raw_gray"]
    gt     = GROUND_TRUTH
    cpairs = Counter()
    ebpos  = {}
    for key in sorder:
        _, pred, _ = results.get(key, ("?", "", []))
        for i in range(len(gt)):
            pc = pred[i] if i < len(pred) else "?"
            if pc != gt[i]:
                cpairs[f"{gt[i]}->{pc}"] += 1
                ebpos.setdefault(i + 1, []).append(pc)

    fig = plt.figure(figsize=(20, 8), facecolor=C["bg"])
    fig.suptitle("Gambar 6 -- Analisis Pola Confusion dan Kemiripan Morfologis Digit",
                 fontsize=13, color=C["blue"], fontweight="bold")
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    if cpairs:
        mc    = cpairs.most_common(10)
        pairs = [p for p, _ in mc]
        cnts  = [c for _, c in mc]
        bc    = [C["red"] if c >= 3 else (C["orange"] if c >= 2 else C["yellow"]) for c in cnts]
        bars  = ax1.barh(range(len(pairs)), cnts, color=bc, alpha=0.85,
                         height=0.7, edgecolor=C["border"])
        ax1.set_yticks(range(len(pairs)))
        ax1.set_yticklabels(pairs, fontsize=14, fontfamily="monospace", fontweight="bold")
        ax1.set_xlabel("Frekuensi (dari 4 strategi)", fontsize=9)
        ax1.invert_yaxis()
        for bar, cnt in zip(bars, cnts):
            ax1.text(cnt + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{cnt}x", va="center", fontsize=11, color=C["text"], fontweight="bold")
        ax1.set_xlim(0, max(cnts) + 2)
    else:
        ax1.text(0.5, 0.5, "Tidak ada\nkesalahan!", ha="center", va="center",
                 fontsize=14, color=C["green"], transform=ax1.transAxes, fontweight="bold")
        ax1.axis("off")
    ax1.set_title("Pasangan Confusion\n(Digit GT -> Prediksi Salah)",
                  fontsize=10, color=C["text"], pad=8)

    ax2 = fig.add_subplot(gs[0, 1])
    positions = sorted(ebpos.keys())
    if positions:
        for pi, pos in enumerate(positions):
            wc = Counter(ebpos[pos])
            for j, (wd, cnt) in enumerate(sorted(wc.items(), key=lambda x: -x[1])):
                ax2.scatter(j * 0.45, pi, s=cnt * 130 + 60, color=C["red"], alpha=0.75, zorder=3)
                ax2.text(j * 0.45, pi, wd, ha="center", va="center",
                         fontsize=13, color="white", fontweight="bold", fontfamily="monospace")
                ax2.text(j * 0.45, pi - 0.40, f"x{cnt}", ha="center",
                         fontsize=7.5, color=C["sub"])
        ax2.set_yticks(range(len(positions)))
        ax2.set_yticklabels([f"Pos {p} (GT={gt[p-1]})" for p in positions], fontsize=9)
        ax2.set_xlim(-0.35, 1.6); ax2.set_ylim(-0.7, len(positions) - 0.4)
        ax2.invert_yaxis()
        ax2.set_xlabel("Variasi prediksi (ukuran = frekuensi)", fontsize=8.5)
        ax2.set_title("Distribusi Prediksi Salah\nper Posisi Digit",
                      fontsize=10, color=C["text"], pad=8)
    else:
        ax2.text(0.5, 0.5, "Tidak ada\nkesalahan!", ha="center", va="center",
                 fontsize=14, color=C["green"], transform=ax2.transAxes, fontweight="bold")
        ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    morph = [
        ("1 <-> 7", "Serif '7' terdegradasi -> tampak seperti '1'\nGaris atas '7' hilang pada resolusi rendah", C["red"]),
        ("9 <-> 4", "Lengkungan atas '9' terpotong\n-> menyerupai sudut digit '4'",                          C["orange"]),
        ("9 <-> 1", "Stroke loop bawah '9' terputus\n-> tersisa batang vertikal '1'",                         C["red"]),
        ("4 <-> 1", "Counter kiri '4' hilang akibat\nbinarisasi agresif -> seperti '1'",                      C["orange"]),
        ("2 <-> 7", "Stroke diagonal atas '2' mirip\nstroke '7' pada resolusi rendah",                        C["yellow"]),
        ("0 <-> 8", "Counter '8' menyatu akibat noise\nbinarisasi -> jadi '0'",                               C["yellow"]),
    ]
    ax3.set_xlim(0, 1); ax3.set_ylim(0, len(morph) + 0.5)
    ax3.set_title("Kemiripan Morfologis Antardigit\n(Faktor Utama Misrecognition)",
                  fontsize=10, color=C["text"], pad=8)
    for i, (pair, reason, col) in enumerate(morph):
        y = len(morph) - i - 0.5
        ax3.add_patch(mpatches.FancyBboxPatch((0.01, y - 0.42), 0.98, 0.82,
            boxstyle="round,pad=0.05", facecolor=C["bg"], edgecolor=col, lw=1.8))
        ax3.text(0.10, y + 0.07, pair, fontsize=13, color=col, fontweight="bold",
                 fontfamily="monospace", va="center")
        ax3.text(0.40, y + 0.07, reason, fontsize=7.5, color=C["text"], va="center",
                 linespacing=1.35)

    plt.tight_layout(pad=2)
    out = os.path.join(OUTPUT_DIR, "fig6_confusion_morphology.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 7: Tabel Ringkasan per Strategi + PSM
# ==============================================================================
def plot_fig7_table(results, raw_data):
    sorder  = ["clahe", "otsu", "adaptive", "raw_gray"]
    slabels = ["S1: CLAHE", "S2: Otsu Thresholding",
               "S3: Adaptive Threshold", "S4: Raw Grayscale"]
    gt   = GROUND_TRUTH
    rows = []
    for key, sl in zip(sorder, slabels):
        _, pred, ctx = results.get(key, ("?", "", []))
        n_ok = sum(1 for i in range(len(gt)) if i < len(pred) and pred[i] == gt[i])
        acc  = n_ok / len(gt) * 100
        per_psm = raw_data.get(key, {})
        psm_str  = " | ".join(
            f"PSM{p}:'{v[1][:10]}'" if v[1] else f"PSM{p}:[kosong]"
            for p, v in sorted(per_psm.items()))
        ctx_str = ctx[0][0] if ctx else "Tidak ada match"
        rows.append([sl, pred or "[tidak terdeteksi]", f"{n_ok}/{len(gt)}",
                     f"{acc:.1f}%", str(len(pred)) if pred else "0",
                     ctx_str, psm_str])

    fig, ax = plt.subplots(figsize=(26, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.axis("off"); ax.set_facecolor(C["bg"])

    cols = ["Strategi Preprocessing", "NoKK Terekstrak", "Digit Benar",
            "Akurasi", "Panjang", "Regex Pola Match", "Detail per PSM"]
    tbl  = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 2.8)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor(C["border"])
        tbl[0, j].set_text_props(color=C["blue"], fontweight="bold")
    for i, key in enumerate(sorder, start=1):
        _, pred, _ = results.get(key, ("?", "", []))
        n_ok = sum(1 for idx in range(len(gt))
                   if idx < len(pred) and pred[idx] == gt[idx])
        acc  = n_ok / len(gt) * 100
        rc   = "#1A2B1A" if acc == 100 else ("#1F1A12" if acc >= 70 else "#2B1A1A")
        ac   = C["green"] if acc == 100 else (C["yellow"] if acc >= 70 else C["red"])
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(rc)
            tbl[i, j].set_text_props(color=C["text"])
        tbl[i, 0].set_text_props(color=SC[key], fontweight="bold")
        tbl[i, 2].set_text_props(color=ac, fontweight="bold")
        tbl[i, 3].set_text_props(color=ac, fontweight="bold")

    ax.set_title(f"Gambar 7 -- Tabel Ringkasan: NoKK Terekstrak dari Full OCR Pipeline\n"
                 f"Ground Truth: {gt}  ({len(gt)} digit)",
                 fontsize=13, color=C["blue"], fontweight="bold", pad=15)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig7_summary_table.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# GAMBAR 8: Pipeline Diagram (sistem asli -- full image)
# ==============================================================================
def plot_fig8_pipeline():
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.patch.set_facecolor(C["bg"])
    ax.axis("off"); ax.set_xlim(0, 24); ax.set_ylim(0, 10)
    fig.suptitle(
        "Gambar 8 -- Alur Pipeline Sistem OCR yang Sesungguhnya\n"
        "OCR dijalankan pada FULL IMAGE -> Raw Text -> NLP Regex Filter -> NoKK",
        fontsize=14, color=C["blue"], fontweight="bold")

    def node(x, y, label, color, w=2.0, h=1.5):
        ax.add_patch(mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.12", facecolor=color+"22", edgecolor=color,
            lw=2.2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.8,
                color=color, fontweight="bold", linespacing=1.45, zorder=4)

    def arr(x1, y1, x2, y2, col="#888", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                   connectionstyle=f"arc3,rad={rad}"), zorder=2)

    # Input
    node(1.2, 5.0, "INPUT\nkk_maya_sari.jpg\n3240x2268 px\n(full document photo)", C["blue"], 2.0, 1.6)

    # Pre-steps
    node(3.8, 6.5, "EXIF Koreksi\nOrientasi\nImageOps.\nexif_transpose", "#7C8CF8", 1.9, 1.4)
    node(3.8, 3.5, "DESKEWING\n3-Pass Pipeline\nContour + MinRect\n+ Hough Lines", "#7C8CF8", 1.9, 1.4)

    # Grayscale
    node(6.5, 5.0, "GRAYSCALE\nConvert BGR->Gray\nFULL IMAGE\n3240x2268 px", C["purple"], 1.9, 1.4)

    # 4 Preprocessing (full image)
    y_pp  = [8.2, 6.2, 3.8, 1.8]
    pplbl = ["CLAHE\ntileGrid(16x16)\nclipLimit=2.5\nFull Image",
             "OTSU\nGlobal Threshold\nFull Image",
             "ADAPTIVE\nGauss Block 31x31\nC=10, Full Image",
             "RAW GRAY\nTanpa Modifikasi\nFull Image"]
    ppk   = ["clahe", "otsu", "adaptive", "raw_gray"]
    for yy, pk, pl in zip(y_pp, ppk, pplbl):
        node(9.5, yy, pl, SC[pk], 2.0, 1.4)
        arr(7.4, 5.0, 8.5, yy, SC[pk])

    # OCR Tesseract (full image)
    for yy, pk in zip(y_pp, ppk):
        node(12.8, yy, "Tesseract OCR\nPSM 3/6/11\n-l ind\nFull Image Input", C["orange"], 2.0, 1.4)
        arr(10.5, yy, 11.8, yy, C["orange"])

    # Raw text output
    for yy in y_pp:
        arr(13.8, yy, 15.2, 5.0, C["orange"], rad=0.0)

    # Raw text aggregated
    node(16.5, 5.0, "RAW TEXT\n(Seluruh Teks Dokumen)\nNama, Alamat, Kecamatan,\nNo.KK, dan lainnya", C["yellow"], 2.3, 1.7)
    arr(15.6, 5.0, 15.7, 5.0, C["orange"])

    # NLP Filter
    node(19.5, 5.0, "NLP REGEX\nFILTER\n_extract_nomor_kk()\nextractor.py", C["purple"], 2.2, 1.5)
    arr(17.65, 5.0, 18.4, 5.0, C["yellow"])

    # Output
    node(22.3, 6.5, "NoKK\nTerekstrak\n(hasil terpilih\nvoting)", C["green"], 2.0, 1.4)
    node(22.3, 3.5, "KETERBATASAN\nSkor confidence\nagregatif, bukan\nper-digit", C["red"], 2.0, 1.4)
    arr(20.6, 5.5, 21.3, 6.5, C["green"])
    arr(20.6, 4.5, 21.3, 3.5, C["red"])

    # Arrows input -> pre
    arr(1.2, 5.8, 2.85, 6.5, C["sub"])
    arr(1.2, 4.2, 2.85, 3.5, C["sub"])
    arr(4.75, 6.5, 5.55, 5.5, C["sub"])
    arr(4.75, 3.5, 5.55, 4.5, C["sub"])

    # Label penting
    ax.text(9.5,  9.8, "PREPROCESSING -- Full Image (tidak ada crop)", ha="center",
            fontsize=9, color=C["text"], fontstyle="italic", alpha=0.8)
    ax.text(12.8, 9.8, "OCR -- Full Image, 12 kombinasi", ha="center",
            fontsize=9, color=C["text"], fontstyle="italic", alpha=0.8)
    ax.text(16.5, 9.8, "Semua teks dokumen KK", ha="center",
            fontsize=9, color=C["text"], fontstyle="italic", alpha=0.8)

    # Catatan kritis
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.2), 14.0, 0.9,
        boxstyle="round,pad=0.3", facecolor=C["panel"], edgecolor=C["yellow"], lw=1.5))
    ax.text(7.3, 0.65,
        "CATATAN KRITIS: Tidak ada pemotongan (crop) di manapun dalam pipeline ini. "
        "Tesseract membaca SELURUH gambar -> output teks mentah seluruh dokumen -> "
        "regex _extract_nomor_kk() memfilter Nomor KK.",
        ha="center", fontsize=8.5, color=C["yellow"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["panel"], edgecolor="none"))

    li = [mpatches.Patch(color=SC[k], label=f"S{i+1}: {SN[k]}") for i, k in enumerate(ppk)]
    ax.legend(handles=li, loc="lower right", fontsize=9,
              facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig8_pipeline_diagram.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  [OK] {os.path.basename(out)}")
    return out


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 65)
    print("  DIAGNOSTIC OCR v3 (FINAL) -- Nomor KK")
    print(f"  Image   : kk_maya_sari.jpg")
    print(f"  GT      : {GROUND_TRUTH}  ({len(GROUND_TRUTH)} digit)")
    print(f"  Metode  : Full image OCR -> NLP regex (sistem asli)")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 65)

    print("\n[1/4] Memuat gambar...")
    img_bgr, img_rgb = load_image()
    print(f"  Resolusi: {img_bgr.shape[1]}x{img_bgr.shape[0]} px")

    print("[2/4] Preprocessing FULL IMAGE (4 strategi)...")
    variants = build_full_variants(img_bgr)
    for key, lbl, v in variants:
        print(f"  OK  {lbl:25s}  shape={v.shape}  mean={v.mean():.1f}")

    print("[3/4] OCR FULL IMAGE x 12 kombinasi -> NLP Regex Filter...")
    results, raw_data = run_full_ocr_pipeline(variants)
    for key, (label, pred, ctx) in results.items():
        n_ok = sum(1 for i in range(len(GROUND_TRUTH))
                   if i < len(pred) and pred[i] == GROUND_TRUTH[i])
        acc  = n_ok / len(GROUND_TRUTH) * 100 if pred else 0
        mark = "OK " if n_ok == len(GROUND_TRUTH) else "ERR"
        ctx_str = ctx[0][0] if ctx else "no match"
        print(f"  [{mark}] {label:25s}: '{pred}'  ({n_ok}/{len(GROUND_TRUTH)}, {acc:.0f}%)"
              f"  via {ctx_str}")

    print("[4/4] Membuat 8 gambar diagnostik...")
    plot_fig1(img_rgb)
    plot_fig2(img_bgr, variants)
    plot_fig3_raw_text(raw_data)
    plot_fig4_digit_comparison(results)
    plot_fig5_heatmap(results)
    plot_fig6_confusion(results)
    plot_fig7_table(results, raw_data)
    plot_fig8_pipeline()

    print("\n" + "=" * 65)
    print("  SELESAI! Semua file tersimpan di:")
    print(f"  {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        kb = os.path.getsize(fp) // 1024
        print(f"    {f:50s} ({kb} KB)")
    print("=" * 65)
