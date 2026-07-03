"""
preprocessing.py — Pemrosesan Citra dan Ekstraksi Teks OCR
============================================================
Menangani:
  - Koreksi orientasi portrait → landscape (Modul 1D)
  - Deskewing otomatis via Hough Line Transform (Modul 1D)
  - 5–6 strategi preprocessing citra: CLAHE, Otsu, Adaptive, Raw gray,
    Shadow, Gamma (Modul 3)
  - OCR 15–18 kombinasi (5–6 preprocessing × PSM 3/6/11) + voting skor (Modul 4)
  - Skor kualitas OCR berbobot v3 (_ocr_quality_score)
  - Pembersihan teks OCR (clean_text)

Optimasi v5.7-Perf:
  - 12 kombinasi OCR per file dijalankan paralel via ThreadPoolExecutor
    (4 worker — sesuaikan OCR_WORKERS untuk hardware yang berbeda)

Perbaikan v5.7.3:
  - Tambah Pass 0 deteksi kontur kartu (card contour detection) sebelum
    MinAreaRect. Mendeteksi persegi panjang terbesar (tepi kartu ID) untuk
    mengoreksi kemiringan besar (>10°) dari foto kamera HP. Sangat efektif
    untuk background gelap seperti gambar yang dikirim user.

Perbaikan v5.7.5:
  - Re-aktifkan Pass 0 (CardContour) dengan validasi lebih ketat:
      * Filter aspect ratio: hanya terima kontur dengan AR = 1.2–2.2
        (sesuai dimensi kartu ID standar 85.6mm × 54mm ≈ 1.58:1)
      * Filter area minimum naik dari 5% → 10% gambar
      * Filter rectangularity: area kontur / area minAreaRect ≥ 0.5
        (menolak kontur tangan/jari yang tidak persegi panjang)
      * Hanya aktif untuk sudut >10° (kemiringan kecil tetap ditangani Pass 1+2)
  - Perbaiki ekstraksi alamat SIM: format bernomor urut dan NO./RT. di akhir baris

Perbaikan v5.8.0 — Koreksi Bayangan (Shadow Correction):
  - Tambah fungsi remove_shadow() berbasis morphological closing untuk
    mengoreksi pencahayaan tidak merata (bayangan separuh, vignette, gradien).
    Khusus mengatasi masalah foto SIM yang tertutupi bayangan di separuh gambar.
  - Tambah fungsi has_uneven_illumination() untuk deteksi otomatis apakah
    gambar memiliki bayangan/pencahayaan tidak merata sebelum koreksi.
  - Tambah varian preprocessing ke-5 "shadow" — gambar yang sudah dikoreksi
    bayangan digunakan sebagai input tambahan untuk OCR voting. Total kombinasi
    naik dari 12 (4×3) menjadi 15 (5×3).
  - CLAHE tileGridSize dinaikkan dari (8,8) → (16,16) agar lebih efektif untuk
    gambar beresolusi tinggi dari kamera HP.

Perbaikan v5.8.1 — Perbaikan Deteksi Bayangan SIM:
  - Turunkan threshold has_uneven_illumination() dari 55.0 → 30.0 agar
    bayangan halus/gradual juga terdeteksi.
  - Tambah deteksi quadrant: bagi gambar 4 kuadran, bandingkan max-min
    rata-rata intensitas (threshold 40.0). Menangkap bayangan diagonal/pojok
    yang lolos deteksi separuh (kiri-kanan / atas-bawah).
  - Kernel remove_shadow() sekarang proporsional resolusi gambar (~15%
    dimensi terpanjang, min 51, max 255). Lebih efektif untuk bayangan
    gradual pada foto HD dari kamera HP.
  - Tambah fallback di extract_text(): jika skor OCR terbaik < 50 dan
    varian shadow belum aktif, paksa shadow correction + OCR ulang.
  - Tambah varian gamma correction untuk gambar sangat gelap (mean < 100).
"""
import re, logging
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytesseract

from config import (
    PORTRAIT_THRESHOLD, SQUARE_THRESHOLD, ROT180_SCORE_MARGIN,
    MIN_IMG_DIMENSION, UPSCALE_TARGET,
)

log = logging.getLogger("otomatisasi_dokumen")


# =============================================================================
# MODULE 3A — KOREKSI BAYANGAN (v5.8.0)
#
# Mengatasi masalah pencahayaan tidak merata pada foto SIM/KTP/KK:
#   - Bayangan separuh (setengah kartu gelap, setengah terang)
#   - Vignette (sudut-sudut gambar lebih gelap dari tengah)
#   - Gradien pencahayaan (satu sisi lebih terang dari sisi lain)
#
# Teknik: Background subtraction via Morphological Closing
#   1. Estimasi latar (background) dengan morphological closing kernel besar
#      (61×61 piksel ≈ 10-15% lebar kartu) — kernel besar "melompati" teks/detail
#      dan hanya menangkap variasi pencahayaan berskala besar.
#   2. Bagi gambar asli dengan estimasi background → normalisasi pencahayaan.
#   3. Hasil: gambar dengan pencahayaan merata, teks tetap terbaca.
# =============================================================================

def has_uneven_illumination(gray, threshold: float = 30.0) -> bool:
    """
    Deteksi apakah gambar memiliki pencahayaan tidak merata (bayangan/gradien).

    Perbaikan v5.8.1: threshold diturunkan dari 55.0 → 30.0 dan ditambah
    deteksi quadrant agar bayangan halus/diagonal juga terdeteksi.

    Tiga level deteksi:
      1. Separuh kiri vs kanan DAN atas vs bawah (threshold 30.0)
      2. Quadrant: 4 kuadran, max-min rata-rata (threshold 40.0)
         Menangkap bayangan diagonal yang lolos deteksi separuh.
      3. Standar deviasi blok: jika variasi std dev antar blok tinggi,
         gambar memiliki pencahayaan tidak merata.

    Args:
        gray      (np.ndarray): Gambar grayscale 2D.
        threshold (float)     : Selisih intensitas minimum (0-255) untuk
                                dianggap "tidak merata". Default 30.0.

    Returns:
        bool: True jika pencahayaan tidak merata dan perlu dikoreksi.
    """
    h, w = gray.shape

    # ── Level 1: deteksi separuh (kiri-kanan, atas-bawah) ────────────────
    left_mean  = float(gray[:, :w // 2].mean())
    right_mean = float(gray[:, w // 2:].mean())
    top_mean   = float(gray[:h // 2, :].mean())
    bot_mean   = float(gray[h // 2:, :].mean())
    diff_lr = abs(left_mean - right_mean)
    diff_tb = abs(top_mean  - bot_mean)
    uneven = (diff_lr > threshold or diff_tb > threshold)

    # ── Level 2: deteksi quadrant (bayangan diagonal/pojok) ──────────────
    if not uneven:
        q_tl = float(gray[:h // 2, :w // 2].mean())   # top-left
        q_tr = float(gray[:h // 2, w // 2:].mean())   # top-right
        q_bl = float(gray[h // 2:, :w // 2].mean())   # bottom-left
        q_br = float(gray[h // 2:, w // 2:].mean())   # bottom-right
        q_vals = [q_tl, q_tr, q_bl, q_br]
        q_range = max(q_vals) - min(q_vals)
        if q_range > 40.0:
            uneven = True
            log.debug(
                f"  ShadowDetect (quadrant): TL={q_tl:.1f} TR={q_tr:.1f} "
                f"BL={q_bl:.1f} BR={q_br:.1f} range={q_range:.1f} → tidak merata"
            )

    if uneven:
        log.debug(
            f"  ShadowDetect: L={left_mean:.1f} R={right_mean:.1f} "
            f"T={top_mean:.1f} B={bot_mean:.1f} "
            f"(ΔLR={diff_lr:.1f}, ΔTB={diff_tb:.1f}) → pencahayaan tidak merata"
        )
    return uneven


def remove_shadow(gray, kernel_pct: float = 0.15, norm_target: float = 220.0) -> "np.ndarray":
    """
    Koreksi pencahayaan tidak merata (bayangan separuh, vignette, gradien)
    menggunakan teknik background subtraction via Gaussian Blur.

    Perbaikan v5.8.2 — Overhaul total dari versi sebelumnya:
      - Ganti Morphological Closing → Gaussian Blur. Gaussian blur lebih
        efektif mengestimasi background dari bayangan gradual/halus yang
        umum pada foto dokumen dari kamera HP.
      - Kernel diperbesar dari 5% → 15% dimensi terpanjang (cap 255).
        Kernel harus JAUH lebih besar dari teks agar hanya menangkap
        variasi pencahayaan, bukan detail teks.
      - Normalisasi target dinaikkan dari 128 → 220 agar area yang tadinya
        gelap (bayangan) menjadi cukup terang untuk Tesseract.

    Algoritma:
      1. Estimasi background dengan Gaussian blur kernel sangat besar.
         Blur besar "meratakan" semua detail lokal (teks, garis) sehingga
         yang tersisa hanya variasi pencahayaan berskala besar.
      2. Normalisasi: gray / background × norm_target.
         Area gelap (bayangan) dinormalisasi naik, area terang turun.
      3. Clip ke [0, 255] dan konversi ke uint8.

    Args:
        gray        (np.ndarray): Gambar grayscale 2D (uint8).
        kernel_pct  (float)     : Ukuran kernel sebagai fraksi dimensi terpanjang.
                                  Default 0.15 (15%).
        norm_target (float)     : Target intensitas normalisasi. Default 220.0.
                                  Nilai lebih tinggi = hasil lebih terang.

    Returns:
        np.ndarray: Gambar grayscale terkoreksi (uint8), pencahayaan merata.
    """
    h, w = gray.shape
    # Kernel proporsional terhadap resolusi (v5.8.2)
    # 15% dimensi terpanjang, min 51, max 255
    ksize = max(51, min(255, int(max(h, w) * kernel_pct)))
    # Pastikan ganjil (syarat GaussianBlur)
    if ksize % 2 == 0:
        ksize += 1

    # Estimasi background dengan Gaussian blur besar
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Normalisasi: hilangkan variasi background
    gray_f = gray.astype(np.float32)
    bg_f   = background.astype(np.float32)
    corrected = (gray_f / (bg_f + 1e-6)) * norm_target
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    log.debug(
        f"  ShadowRemove: GaussianBlur k={ksize}, "
        f"norm_target={norm_target:.0f}, selesai"
    )
    return corrected


def _estimate_skew_card_contour(gray):
    """
    Estimasi sudut kemiringan via deteksi kontur kartu (Pass 0) — v5.7.5.

    Mendeteksi persegi panjang terbesar di gambar (tepi kartu ID/KTP/SIM/KK)
    menggunakan aproksimasi kontur. Sangat efektif untuk foto dengan background
    kontras (gelap/terang) dan kemiringan besar (10° s/d 60°).

    Perbaikan v5.7.5 — validasi lebih ketat untuk menghindari false positive
    akibat tangan/jari yang menutupi tepi kartu:
      1. Filter aspect ratio (AR) = 1.2–2.2 (sesuai kartu ID standar ≈ 1.58:1)
      2. Area minimum naik ke 10% gambar (bukan 5%)
      3. Rectangularity check: area kontur / area minAreaRect ≥ 0.5
         (menolak kontur tidak persegi yang biasanya tangan/background)

    Strategi:
      1. Edge detection dengan Canny (tiga threshold untuk robustness)
      2. Dilasi ringan agar kontur kartu tidak terputus
      3. Temukan kontur 4-8 sisi terbesar yang lolos filter AR + rectangularity
      4. Ambil sudut dari sisi terpanjang kontur tersebut

    Returns:
        float or None: Sudut kemiringan (derajat, positif = tilt CW). None jika gagal.
    """
    import numpy as np

    h, w = gray.shape
    img_area = h * w

    # Blur ringan untuk mengurangi noise sebelum edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    best_angle = None
    best_area  = 0

    # Coba beberapa threshold Canny untuk robustness
    for lo, hi in [(30, 100), (50, 150), (10, 60)]:
        edges = cv2.Canny(blurred, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Kartu harus menempati 10%–88% area gambar (lebih ketat dari sebelumnya)
            if area < img_area * 0.10 or area > img_area * 0.88:
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Hanya terima kontur 4-8 sisi (persegi panjang / sedikit melengkung)
            if not (4 <= len(approx) <= 8):
                continue

            # ── Filter baru v5.7.5 ────────────────────────────────────────────
            rect   = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw == 0 or rh == 0:
                continue

            # 1. Filter aspect ratio: kartu ID standar ≈ 1.58:1
            long_side  = max(rw, rh)
            short_side = min(rw, rh)
            ar = long_side / short_side
            if not (1.2 <= ar <= 2.2):
                continue   # bukan proporsi kartu ID

            # 2. Filter rectangularity: area kontur / area minAreaRect
            rect_area = rw * rh
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < 0.35:
                continue   # terlalu tidak persegi — kemungkinan tangan/latar
            # ─────────────────────────────────────────────────────────────────

            if area > best_area:
                best_area = area
                angle  = rect[2]
                # Normalisasi sudut ke sudut koreksi yang dibutuhkan:
                # minAreaRect angle: [-90, 0). rw adalah sisi pertama.
                # Kita ingin sudut sisi TERPANJANG terhadap horizontal.
                if rw < rh:
                    # sisi panjang = rh, sudut sisi panjang = angle + 90
                    norm_angle = angle + 90.0
                else:
                    norm_angle = angle
                # norm_angle sekarang ≈ sudut dari horizontal (bisa -90 s/d +90)
                # Normalisasi ke -90..+90
                while norm_angle > 90.0:  norm_angle -= 180.0
                while norm_angle < -90.0: norm_angle += 180.0
                # Koreksi yang dibutuhkan: kebalikan dari kemiringan
                best_angle = -norm_angle

        if best_angle is not None:
            break

    if best_angle is None:
        return None

    # Hanya kembalikan jika kemiringan signifikan dan masuk akal.
    # Abaikan jika sudut sangat dekat 0 (sudah lurus) atau mendekati +/-90
    # (kemungkinan artefak kartu portrait yang sudah di-transpose EXIF).
    if abs(best_angle) < 2.0 or abs(best_angle) > 55.0:
        return None

    log.debug(f"  Deskew Pass 0 (CardContour): kandidat sudut={best_angle:+.1f}°")
    return float(best_angle)


def _estimate_skew_minarearect(gray):
    """
    Estimasi sudut kemiringan dokumen via MinAreaRect pada blob teks.

    Perbaikan v5.7.5-r4: 
    - Menggunakan kernel Elips isotropik (15x15) agar dapat menemukan teks 
      walaupun miring ekstrem (s/d 90 derajat). Kernel horizontal (25x3) lama 
      akan mengacaukan teks vertikal/miring tajam.
    - Menghitung orientasi sisi terpanjang (asumsi teks selalu lebih lebar dari tingginya).

    Returns:
        float or None: Sudut kemiringan (derajat, positif = tilt CW). None jika gagal.
    """
    import numpy as np

    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Kernel isotropik untuk menggabungkan karakter, tidak peduli arahnya
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(binary, kernel)
    
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    
    h, w      = gray.shape
    img_area  = h * w
    skew_list = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.001 or area > img_area * 0.5:
            continue
            
        rect   = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
            
        # Filter kotak yang terlalu persegi (biasanya foto/logo, bukan teks).
        # v5.7.6-fix: Turunkan threshold dari 1.5 → 1.2 agar blob sel tabel KK
        # (rasio 1.0–1.4) tidak diabaikan semuanya, sehingga estimasi sudut
        # tidak bergantung hanya pada noise kecil.
        aspect_ratio = max(rw, rh) / min(rw, rh)
        if aspect_ratio < 1.2:
            continue
            
        angle = rect[2]
        # minAreaRect OpenCV (v4.5+) convention:
        # Sudut terhadap sumbu X positif ke sisi pertama (panjang = rw).
        # Kita butuh sudut sisi TERPANJANG terhadap horizontal.
        if rw < rh:
            angle = angle + 90.0

        # Normalisasi ke rentang -90 s/d +90
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        # minAreaRect angle: negatif = miring CCW.
        # skew = -angle agar koreksi = rotasi berlawanan arah.
        skew = -angle

        # Filter: abaikan sudut yang sangat kecil (sudah lurus) atau
        # mendekati ±90° (artefak kontur vertikal pada kartu horizontal).
        # Gambar KTP/KK yang sudah landscape menghasilkan banyak blob dengan
        # sudut ~0° atau ~±90° yang tidak boleh dianggap sebagai kemiringan.
        if abs(skew) < 1.0 or abs(skew) > 75.0:
            continue
        # Abaikan sudut dekat ±90° (terlalu ambigius antara portrait vs miring)
        if abs(abs(skew) - 90.0) < 10.0:
            continue

        skew_list.append(skew)

    if not skew_list:
        return None

    # Guard minimum sample: median dari 1–2 blob sangat rentan noise.
    # Jika sampel terlalu sedikit, lebih aman tidak mengoreksi apapun.
    if len(skew_list) < 3:
        log.debug(f"  Deskew Pass 1 (MinAreaRect): sampel terlalu sedikit ({len(skew_list)}), skip")
        return None

    best_angle = float(np.median(skew_list))

    # Tolak jika median sangat kecil — gambar sudah cukup lurus
    if abs(best_angle) < 1.5:
        return None

    log.debug(f"  Deskew Pass 1 (MinAreaRect): kandidat sudut={best_angle:+.1f}° dari {len(skew_list)} blob teks")
    return best_angle


def _estimate_skew_hough(gray, img_width):
    """
    Estimasi sudut kemiringan via HoughLinesP.

    Lebih presisi untuk sudut kecil di bawah 10 derajat.
    Digunakan sebagai Pass 2 (refinement) setelah MinAreaRect.

    Returns:
        float or None: Sudut kemiringan (derajat, positif = tilt CW). None jika gagal.
    """
    import numpy as np

    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=img_width // 6, maxLineGap=20,
    )
    if lines is None:
        return None
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Batasi pencarian ke [-10.0, 10.0] karena Pass 2 hanya
            # untuk refinement sudut kecil. Garis di luar rentang ini
            # biasanya pinggiran kartu/noise, bukan baris teks horizontal.
            if -10.0 < a < 10.0:
                angles.append(a)
    if not angles:
        return None
    # Guard minimum sample: median dari < 5 garis tidak cukup representatif.
    if len(angles) < 5:
        log.debug(f"  Deskew Pass 2 (HoughLinesP): sampel terlalu sedikit ({len(angles)}), skip")
        return None
    median_angle = float(np.median(angles))
    # Tolak jika sudut sangat kecil — tidak perlu koreksi, gambar sudah lurus.
    # Threshold dinaikkan ke 0.8° untuk menghindari koreksi trivial yang
    # justru memutar gambar horizontal/vertikal yang sudah benar.
    if abs(median_angle) < 0.8:
        return None
    return median_angle


def _estimate_skew_projection(gray):
    """
    Estimasi sudut kemiringan via HoughLines + Projection Profile (v5.7.5-r3).

    Perbaikan kritis dibanding versi sebelumnya:
    1. CROP: buang 10% atas dan 20% bawah gambar sebelum analisis.
       Untuk foto kartu dengan sisir/comb di bawah, sisir menciptakan
       garis horizontal yang membuat row variance tinggi di 0 deg --
       menyebabkan fungsi selalu return None karena 0 deg dianggap best.
    2. HoughLines sebagai primary detector: cari sudut dominan dari garis-
       garis yang terdeteksi di Canny image. Jauh lebih robust dan cepat
       dari scan exhaustif Projection untuk gambar real.
    3. Projection Profile sebagai verifikasi: konfirmasi dan perbaiki halus
       sudut yang ditemukan HoughLines (+/-10 deg di sekitar kandidat).

    Returns:
        float or None: Sudut koreksi terbaik, atau None jika tidak signifikan.
    """
    import numpy as np

    h, w = gray.shape

    # ── Crop: buang 10% atas + 20% bawah untuk menghindari noise tepi ──────
    # (sisir, background kosong, jari yang memegang kartu)
    top    = max(0, int(h * 0.10))
    bottom = min(h, int(h * 0.80))
    left   = max(0, int(w * 0.03))
    right  = min(w, int(w * 0.97))
    cropped = gray[top:bottom, left:right]

    # Downscale ke 600px untuk performa
    ch, cw = cropped.shape
    scale  = min(1.0, 600.0 / max(ch, cw))
    if scale < 0.99:
        small = cv2.resize(cropped, (int(cw * scale), int(ch * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = cropped.copy()

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    sh, sw = edges.shape
    cx, cy = sw / 2.0, sh / 2.0

    # ── Pass A: HoughLines — cari sudut dominan ──────────────────────────────
    hough_candidate = None
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
    if lines is not None and len(lines) >= 5:
        angles = []
        for ln in lines:
            theta = ln[0][1]
            angle = np.degrees(theta) - 90.0
            # Normalisasi ke -90 s/d +90
            while angle < -90.0: angle += 180.0
            while angle > 90.0:  angle -= 180.0
            angles.append(angle)

        angles = np.array(angles)
        # Histogram dengan 36 bin (5 deg per bin)
        hist, bins = np.histogram(angles, bins=36, range=(-90.0, 90.0))
        peak_idx   = int(np.argmax(hist))
        dominant   = float((bins[peak_idx] + bins[peak_idx + 1]) / 2.0)

        # Median dari semua sudut dalam 15 deg radius dominan
        near = angles[np.abs(angles - dominant) <= 15.0]
        if len(near) >= 3:
            hough_candidate = float(np.median(near))
            log.debug(f"  Deskew Proj HoughLines: {len(near)} garis, "
                      f"dominan={dominant:.1f}, kandidat={hough_candidate:.1f} deg")

    # ── Pass B: Projection Profile untuk verifikasi/refinement ──────────────
    def _proj_score(angle_deg: float) -> float:
        M   = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rot = cv2.warpAffine(edges, M, (sw, sh),
                             flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(np.sum(rot, axis=1).astype(np.float64)))

    if hough_candidate is not None and abs(hough_candidate) > 2.0:
        # Fine-tune di sekitar kandidat HoughLines (±15 deg, step 1)
        best_angle = hough_candidate
        best_score = _proj_score(hough_candidate)
        lo = max(-75, int(hough_candidate) - 15)
        hi = min(75,  int(hough_candidate) + 16)
        for deg in range(lo, hi):
            s = _proj_score(float(deg))
            if s > best_score:
                best_score = s
                best_angle = float(deg)
        # Very fine ±1.5 step 0.5
        base2 = round(best_angle * 2) / 2.0
        for d10 in range(int(base2 * 10) - 15, int(base2 * 10) + 16, 5):
            ang = d10 / 10.0
            s   = _proj_score(ang)
            if s > best_score:
                best_score = s
                best_angle = ang
    else:
        # Tidak ada kandidat HoughLines — full scan (fallback)
        best_angle = 0.0
        best_score = _proj_score(0.0)
        for deg in range(-75, 76, 5):
            s = _proj_score(float(deg))
            if s > best_score:
                best_score = s
                best_angle = float(deg)
        if abs(best_angle) >= 2.0:
            lo = max(-75, int(best_angle) - 8)
            hi = min(75,  int(best_angle) + 9)
            for deg in range(lo, hi):
                s = _proj_score(float(deg))
                if s > best_score:
                    best_score = s
                    best_angle = float(deg)

    # Naikkan threshold minimum ke 3.0° untuk menghindari false positive
    # pada gambar yang sudah horizontal (KTP/KK landscape).
    # Gambar yang sudah lurus bisa mendapatkan best_angle kecil (~1-2°) dari
    # noise garis tepi kartu, yang justru memutar gambar menjadi miring.
    if abs(best_angle) < 3.0 or abs(best_angle) > 75.0:
        return None

    log.debug(f"  Deskew Pass 3 (Proj+Hough): sudut={best_angle:+.1f} deg")
    return float(best_angle)



def deskew_image(img, skip_projection: bool = False):
    """
    Koreksi kemiringan dokumen secara otomatis dengan arsitektur TIGA PASS.

    Perbaikan v5.7.5 -- Re-aktifkan Pass 0 dengan validasi ketat.

    Perbaikan v5.7.6-fix -- Cegah false positive pada gambar landscape:

      Parameter baru `skip_projection` (default False):
        Jika True, Pass 3 (Projection Profile) dilewati sepenuhnya.
        Dipanggil dengan skip_projection=True oleh preprocess_image() ketika
        gambar sudah berhasil diidentifikasi orientasinya (max_score > 0),
        artinya gambar sudah cukup lurus dan Pass 3 tidak diperlukan.
        Pass 3 tetap aktif hanya untuk gambar yang semua skor = 0 (miring besar).

      Threshold Pass 3 dinaikkan dari 8° → 20°:
        Threshold lama (8°) terlalu rendah — garis tabel KK/KTP dapat menipu
        Projection Profile untuk mendeteksi kemiringan palsu 8–15°, lalu
        merotasi gambar landscape yang sudah lurus menjadi miring.
        Threshold 20° memastikan Pass 3 hanya aktif untuk kemiringan besar
        yang benar-benar lolos dari Pass 0+1+2.

      Pass 0 CardContour (v5.7.5 — re-aktif dengan filter ketat):
        Mendeteksi tepi kartu ID sebagai persegi panjang terbesar di gambar.
        Sangat efektif untuk foto dengan background kontras (gelap/terang)
        dan kemiringan besar > 10 derajat seperti foto dari kamera HP.
        Validasi ketat (AR + rectangularity) menghindari false positive tangan.
        Jika berhasil koreksi > 10°, Pass 1 dilewati (sudah cukup presisi).
        Untuk kemiringan ≤ 10°, Pass 0 dilewati dan Pass 1 yang menangani.

      Pass 1 MinAreaRect:
        Mendeteksi dan mengoreksi kemiringan 1 s/d 45 derajat.
        Bekerja pada massa blob teks, bukan deteksi garis individual.
        Dijalankan jika Pass 0 tidak berhasil atau sudut Pass 0 ≤ 10°.

      Pass 2 HoughLinesP:
        Refinement presisi untuk sisa kemiringan kecil setelah Pass 0/1.
        Fallback jika Pass 0 dan Pass 1 tidak mendeteksi apapun.

    Konvensi sudut: positif = tilt CW, koreksi = rotasi -skew.

    Args:
        img           (np.ndarray): Gambar BGR.
        skip_projection (bool)    : Jika True, lewati Pass 3 Projection Profile.
                                    Gunakan True jika orientasi gambar sudah diketahui benar.

    Returns:
        tuple(np.ndarray, float): Gambar terkoreksi dan total sudut koreksi.
    """
    import numpy as np

    def _apply_correction(image, skew_deg):
        ch, cw  = image.shape[:2]
        center  = (cw / 2.0, ch / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, -skew_deg, 1.0)
        return cv2.warpAffine(
            image, rot_mat, (cw, ch),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result     = img
    total_skew = 0.0
    pass0_ok   = False

    # Pass 0: CardContour — Re-aktif (v5.7.5) dengan validasi ketat.
    # Hanya aktif untuk kemiringan >10° (sudut kecil lebih baik ditangani Pass 1+2).
    # Validasi AR + rectangularity menghindari false positive akibat tangan/jari.
    skew0 = _estimate_skew_card_contour(gray)
    if skew0 is not None and abs(skew0) > 10.0:
        result     = _apply_correction(result, skew0)
        total_skew = skew0
        pass0_ok   = True
        log.debug(f"  Deskew Pass 0 (CardContour): {skew0:+.1f} derajat")
    elif skew0 is not None:
        # Sudut terdeteksi tapi ≤ 10° — biarkan Pass 1+2 yang menangani
        log.debug(f"  Deskew Pass 0 (CardContour): sudut kecil {skew0:+.1f}°, serahkan ke Pass 1")

    # Pass 1: MinAreaRect -- berbasis blob teks (jauh lebih robust terhadap tangan/background)
    # Dilewati jika Pass 0 sudah sukses koreksi > 10°
    if not pass0_ok:
        skew1 = _estimate_skew_minarearect(gray)
        if skew1 is not None and abs(skew1) >= 0.5:
            result     = _apply_correction(result, skew1)
            total_skew = skew1
            log.debug(f"  Deskew Pass 1 (MinAreaRect): {skew1:+.1f} derajat")

    # Pass 2: HoughLinesP -- refinement untuk sisa sudut kecil
    gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    skew2 = _estimate_skew_hough(gray2, result.shape[1])
    # Naikkan threshold aplikasi ke 0.8° untuk menghindari koreksi trivial
    # yang justru memperburuk gambar horizontal/vertikal yang sudah benar.
    if skew2 is not None and abs(skew2) >= 0.8:
        result      = _apply_correction(result, skew2)
        total_skew += skew2
        log.debug(f"  Deskew Pass 2 (HoughLinesP): {skew2:+.1f} derajat")

    # Pass 3: Projection Profile -- last resort untuk kemiringan besar yang lolos Pass 0+1+2.
    # Hanya aktif jika:
    #   1. skip_projection=False (dipanggil dari jalur fallback, bukan orientasi normal)
    #   2. Koreksi kumulatif sejauh ini masih < 10° (berarti Pass 0+1 gagal)
    #   3. Threshold aplikasi dinaikkan ke 20.0° (dari 8.0°) untuk menghindari false positive
    #      akibat garis tabel KK/KTP yang menipu Projection Profile.
    if not skip_projection and abs(total_skew) < 10.0:
        gray3 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        skew3 = _estimate_skew_projection(gray3)
        if skew3 is not None and abs(skew3) > 20.0:
            result      = _apply_correction(result, skew3)
            total_skew += skew3
            log.debug(f"  Deskew Pass 3 (Projection): {skew3:+.1f} derajat")

    if abs(total_skew) < 0.5:
        return img, 0.0
    return result, total_skew

def _quick_score(candidate_img, lang="ind") -> int:
    """
    Uji skor OCR dengan 3 PSM berbeda, ambil maksimum.

    Perbaikan v5.7.6-KK: Tambah PSM 3 (fully automatic page segmentation)
    sebagai kandidat ketiga. PSM 6 (uniform text block) tidak cocok untuk
    Kartu Keluarga yang berbentuk TABEL multi-kolom — hanya PSM 3 yang
    dapat mendeteksi keyword KK dari layout tabel secara akurat.
    Tidak mempengaruhi KTP/SIM karena PSM 6 tetap digunakan dan max() diambil.
    """
    try:
        gray      = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_g   = clahe_obj.apply(gray)
        _, otsu_g = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        s1 = _ocr_quality_score(pytesseract.image_to_string(
            clahe_g, config=f"--oem 3 --psm 6 -l {lang}"))   # uniform block (KTP)
        s2 = _ocr_quality_score(pytesseract.image_to_string(
            otsu_g,  config=f"--oem 3 --psm 11 -l {lang}"))  # sparse text
        s3 = _ocr_quality_score(pytesseract.image_to_string(
            clahe_g, config=f"--oem 3 --psm 3 -l {lang}"))   # auto (KK tabel)
        return max(s1, s2, s3)
    except Exception as e:
        log.debug(f"  _quick_score error: {e}")
        return 0

def correct_upside_down(img, lang: str = "ind"):
    """
    Koreksi orientasi 180 derajat untuk gambar yang terbalik.

    Digunakan setelah gambar diluruskan (deskew) dan diposisikan landscape,
    sehingga OCR bisa secara akurat membaca dan menguji skor teks.
    """
    img_180   = cv2.rotate(img, cv2.ROTATE_180)
    score_ori = _quick_score(img, lang)
    score_180 = _quick_score(img_180, lang)
    log.debug(f"  Orientasi OCR -- ori={score_ori}, 180={score_180}")
    if score_180 > score_ori * ROT180_SCORE_MARGIN:
        log.debug("  -> Rotasi 180 dipilih (dokumen terbalik)")
        return img_180, "rotated_180"
    return img, "upright"



def preprocess_image(image_path: str, use_deskew: bool = True) -> list:
    """
    Membaca gambar dan menghasilkan 4 varian preprocessing.

    Urutan proses:
      1. Baca gambar dengan cv2.imread()
      2. Fallback ke Pillow jika path mengandung karakter unicode (Windows)
      3. Resize jika resolusi < 800px (agar Tesseract akurat)
      4. Konversi ke grayscale
      5. Hasilkan 4 varian: CLAHE, Otsu, Adaptive, Raw gray

    Args:
        image_path (str): Path absolut atau relatif ke file gambar.

    Returns:
        list of tuple(label: str, image: np.ndarray):
            - label   : nama strategi ('clahe', 'otsu', 'adaptive', 'raw_gray')
            - image   : array NumPy 2D (grayscale/binary) siap untuk Tesseract
        Mengembalikan list kosong jika file tidak dapat dibaca.
    """
    # PERBAIKAN v5.7.1: Baca via PIL + EXIF transpose terlebih dahulu.
    # Foto dari kamera HP menyimpan rotasi di metadata EXIF -- cv2.imread
    # MENGABAIKAN EXIF ini sehingga gambar tampak miring/terbalik walaupun
    # tampilan preview di Windows terlihat benar.
    img = None
    try:
        from PIL import Image, ImageOps
        import numpy as np
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil)   # Terapkan EXIF rotation/flip otomatis
        pil = pil.convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.debug(f"  PIL/EXIF read gagal: {e} -- coba cv2 langsung")
        img = cv2.imread(image_path)

    if img is None:
        log.error(f"Tidak dapat membaca gambar: {image_path}")
        log.error("  Pastikan file tidak korup dan formatnya didukung.")
        return []

    # ── Koreksi orientasi awal (4 Arah) via OCR ─────────────────────────────
    # Mengecek orientasi awal secara robust untuk menghindari kesalahan pada
    # dokumen vertikal atau terbalik. Kita tes ke-4 orientasi dan pilih skor tertinggi.
    score_0 = _quick_score(img, lang="ind")
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    score_90 = _quick_score(img_90, lang="ind")
    
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    score_180 = _quick_score(img_180, lang="ind")
    
    img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    score_270 = _quick_score(img_270, lang="ind")

    scores = [score_0, score_90, score_180, score_270]
    best_idx = int(np.argmax(scores))
    max_score = scores[best_idx]

    # Perbaikan v5.7.6-KK: Terapkan minimum margin sebelum memutuskan rotasi.
    # Orientasi non-0° hanya dipilih jika skornya JAUH lebih tinggi dari skor 0°.
    # Ini mencegah KK (dan KTP/SIM horizontal) diputar karena perbedaan skor kecil
    # yang disebabkan noise bukan perbedaan orientasi yang nyata.
    # Margin:
    #   - Rotasi 90°/270° (portrait ↔ landscape): butuh margin 1.3× (perubahan besar)
    #   - Rotasi 180° (terbalik): butuh margin 1.15× (lebih toleran, teks terbalik
    #     biasanya selisihnya cukup jelas)
    # Orientasi 0° selalu diterima tanpa margin (default aman).
    MARGIN_90_270 = 1.30   # threshold untuk rotasi 90/270 derajat
    MARGIN_180    = 1.15   # threshold untuk rotasi 180 derajat

    if max_score > 0 and best_idx != 0:
        margin = MARGIN_180 if best_idx == 2 else MARGIN_90_270
        if score_0 > 0 and max_score < score_0 * margin:
            # Selisih skor tidak cukup signifikan — pertahankan orientasi asli
            log.debug(
                f"  Rough Orientation: {best_idx * 90}° ditolak (skor={max_score}, "
                f"0°={score_0}, margin={margin}×) — pertahankan 0°"
            )
            best_idx  = 0
            max_score = score_0

    if max_score > 0:
        imgs = [img, img_90, img_180, img_270]
        img = imgs[best_idx]
        orientation = f"{best_idx * 90}_deg_max"
        log.debug(f"  Rough Orientation: {best_idx * 90}° dipilih (max skor={max_score})")
    else:
        # Semua skor 0 — gambar kemungkinan sangat miring (>20°).
        # Strategi v5.7.5: coba deskew (termasuk Projection Profile) dulu,
        # lalu re-check 4 orientasi pada gambar yang sudah diluruskan.
        orientation  = "0_deg_fallback"
        _pre_deskewed = False
        if use_deskew:
            # skip_projection=False: gambar miring besar, Pass 3 Projection boleh aktif
            img_pre, skew_pre = deskew_image(img, skip_projection=False)
            if abs(skew_pre) >= 5.0:
                # Cek 4 orientasi pada gambar yang sudah diluruskan
                pre_90  = cv2.rotate(img_pre, cv2.ROTATE_90_CLOCKWISE)
                pre_180 = cv2.rotate(img_pre, cv2.ROTATE_180)
                pre_270 = cv2.rotate(img_pre, cv2.ROTATE_90_COUNTERCLOCKWISE)
                sc_pre  = [
                    _quick_score(img_pre, "ind"),
                    _quick_score(pre_90,  "ind"),
                    _quick_score(pre_180, "ind"),
                    _quick_score(pre_270, "ind"),
                ]
                best_pre = int(np.argmax(sc_pre))
                if sc_pre[best_pre] > 0:
                    imgs_pre = [img_pre, pre_90, pre_180, pre_270]
                    img = imgs_pre[best_pre]
                    _pre_deskewed = True
                    orientation = f"pre_deskew_{best_pre * 90}_deg"
                    log.debug(
                        f"  Pre-deskew fallback: koreksi {skew_pre:+.1f}°, "
                        f"orientasi terbaik {best_pre * 90}° (skor={sc_pre[best_pre]})"
                    )
        if not _pre_deskewed:
            log.debug("  Rough Orientation: Semua skor 0, fallback ke 0°")

    # ── Deskewing — koreksi kemiringan sebelum OCR Orientation ─────────
    # Dilewati jika gambar sudah di-deskew pada tahap pre-deskew fallback di atas.
    # _pre_deskewed hanya True jika max_score==0 DAN pre-deskew berhasil memperbaiki orientasi.
    # v5.7.6-fix: kirim skip_projection=True jika gambar sudah teridentifikasi orientasinya
    # (max_score > 0), sehingga Pass 3 Projection tidak merotasi gambar landscape yang sudah lurus.
    _already_deskewed = (max_score == 0) and locals().get('_pre_deskewed', False)
    if use_deskew and not _already_deskewed:
        _skip_proj = (max_score > 0)   # True = gambar sudah lurus, skip Pass 3
        img_before_deskew = img
        score_before_deskew = max_score if max_score > 0 else _quick_score(img, "ind")
        img_deskewed, skew = deskew_image(img, skip_projection=_skip_proj)
        if abs(skew) >= 0.5:
            score_after_deskew = _quick_score(img_deskewed, "ind")
            # Guard untuk false positive deskew: pada foto SIM/KTP yang sudah
            # lurus, kontur kartu atau ilustrasi bisa menghasilkan sudut palsu
            # 10-15 derajat. Jika OCR cepat anjlok, pertahankan gambar awal.
            if (
                score_before_deskew > 0
                and score_after_deskew < score_before_deskew * 0.75
            ):
                img = img_before_deskew
                log.debug(
                    f"  Deskew: koreksi {skew:.1f}° ditolak "
                    f"(skor {score_before_deskew}->{score_after_deskew})"
                )
            else:
                img = img_deskewed
                max_score = max(max_score, score_after_deskew)
                log.debug(
                    f"  Deskew: koreksi {skew:.1f} derajat "
                    f"(skor {score_before_deskew}->{score_after_deskew})"
                )
        else:
            img = img_deskewed

    # ── Koreksi orientasi 180° (Upside-Down check via OCR) ─────────
    img, orientation_ocr = correct_upside_down(img, lang="ind")
    if orientation_ocr == "rotated_180":
        log.debug("  Orientasi OCR: Gambar terbalik, dirotasi 180°")

    # Pastikan resolusi cukup (Tesseract butuh min 300 DPI untuk akurasi optimal)
    h, w = img.shape[:2]
    if max(h, w) < MIN_IMG_DIMENSION:
        scale = UPSCALE_TARGET / max(h, w)
        img   = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── v5.8.0: Koreksi bayangan (shadow correction) ──────────────────────
    # Deteksi pencahayaan tidak merata (bayangan separuh, gradien, vignette).
    # Jika terdeteksi, hasilkan varian ke-5 "shadow" dari gambar yang sudah
    # dikoreksi. Varian ini sangat membantu untuk foto SIM/KTP dengan bayangan.
    if has_uneven_illumination(gray):
        gray_shadow = remove_shadow(gray)
        log.debug("  ShadowCorrection: varian 'shadow' diaktifkan")
    else:
        gray_shadow = None

    # A. CLAHE — tileGridSize dinaikkan (8,8)→(16,16) untuk gambar HD dari HP
    clahe_obj  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    clahe_img  = clahe_obj.apply(gray)

    # B. Otsu global thresholding
    _, otsu_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # C. Adaptive Gaussian (setelah denoising untuk kurangi false threshold)
    denoised    = cv2.fastNlMeansDenoising(gray, h=10)
    adaptive_img = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )

    variants = [
        ("clahe",    clahe_img),
        ("otsu",     otsu_img),
        ("adaptive", adaptive_img),
        ("raw_gray", gray),
    ]

    # D. Shadow variants — MULTIPLE variants dari gambar shadow-corrected (v5.8.2)
    # Satu varian shadow saja tidak cukup — Tesseract sensitif terhadap
    # preprocessing, sehingga kita perlu mencoba beberapa teknik pada gambar
    # yang sudah dikoreksi bayangan.
    if gray_shadow is not None:
        # CLAHE khusus shadow: clipLimit LEBIH TINGGI (3.0) dari standar (2.5)
        # karena gambar shadow-corrected cenderung "flat" (kontras rendah)
        # setelah normalisasi, sehingga butuh enhancement kontras lebih agresif.
        clahe_shadow     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        clahe_shadow_hi  = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

        # D1. Shadow + CLAHE 3.0 (kontras lokal optimal)
        variants.append(("shadow", clahe_shadow.apply(gray_shadow)))

        # D2. Shadow + CLAHE 5.0 (kontras sangat agresif)
        variants.append(("shadow_hi", clahe_shadow_hi.apply(gray_shadow)))

        # D3. Shadow + Otsu (binarisasi global)
        _, shadow_otsu = cv2.threshold(
            gray_shadow, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        variants.append(("shadow_otsu", shadow_otsu))

        # D4. Shadow + Adaptive threshold (binarisasi lokal)
        shadow_denoised = cv2.fastNlMeansDenoising(gray_shadow, h=10)
        shadow_adaptive = cv2.adaptiveThreshold(
            shadow_denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        variants.append(("shadow_adapt", shadow_adaptive))

        # D5. Shadow + equalizeHist (histogram stretching penuh)
        shadow_eq = cv2.equalizeHist(gray_shadow)
        variants.append(("shadow_eq", shadow_eq))

    # E. Gamma correction — untuk gambar sangat gelap (v5.8.1)
    # Gambar dengan bayangan parah sering memiliki mean intensity rendah.
    # Gamma correction mencerahkan area gelap tanpa over-expose area terang.
    mean_intensity = float(gray.mean())
    if mean_intensity < 100:
        gamma = 2.0 if mean_intensity < 70 else 1.5
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        gamma_img = cv2.LUT(gray, table)
        # Terapkan CLAHE pada gambar gamma-corrected
        gamma_clahe = clahe_obj.apply(gamma_img)
        variants.append(("gamma", gamma_clahe))
        log.debug(
            f"  GammaCorrection: mean={mean_intensity:.1f}, gamma={gamma}, "
            f"varian 'gamma' diaktifkan"
        )

    return variants


# =============================================================================
# MODULE 4 — EKSTRAKSI TEKS (OCR)
#
# Menjalankan Tesseract OCR pada semua kombinasi preprocessing × PSM.
# Total: 4–5 preprocessing × 3 PSM = 12–15 kombinasi per dokumen.
#   - 12 kombinasi (4×3) untuk foto dengan pencahayaan normal
#   - 15 kombinasi (5×3) untuk foto dengan bayangan/pencahayaan tidak merata
#     (varian "shadow" ditambahkan otomatis oleh preprocess_image)
#
# PSM (Page Segmentation Mode) yang digunakan:
#   PSM 6  — Asumsikan satu blok teks seragam. Terbaik untuk KTP/KK terstruktur.
#   PSM 11 — Sparse text (cari teks di mana saja). Terbaik untuk layout bebas.
#   PSM 3  — Fully automatic (default Tesseract). Fallback umum.
#
# Pemilihan teks terbaik:
#   Skor = jumlah karakter alfanumerik (A-Z, a-z, 0-9) dalam hasil OCR.
#   Teks dengan skor tertinggi dipilih sebagai "best_text".
#   Semua hasil disimpan dalam "all_texts" untuk voting nama (Module 6).
#
# (Rusli et al., 2020, Sec. III-B — Tesseract sebagai OCR engine terbaik,
#  F-score 0.46 total area, 0.89 untuk scanner)
# =============================================================================
# Jumlah worker thread untuk paralel OCR (12 kombinasi per file)
# Sesuaikan dengan jumlah core CPU — 4 untuk laptop modern, 6-8 untuk workstation
OCR_WORKERS = 4


def extract_text(image_path: str, lang: str = "ind",
                use_deskew: bool = True) -> tuple:
    """
    Menjalankan OCR dengan 12–18 kombinasi secara PARALEL.

    Optimasi v5.7-Perf: Semua kombinasi dijalankan bersamaan menggunakan
    ThreadPoolExecutor (OCR_WORKERS thread). Ini mengurangi waktu per file
    dari ~12x ke ~1-2x waktu OCR tunggal pada hardware multi-core.

    Perbaikan v5.8.1: Jika skor OCR terbaik sangat rendah (< 50) dan varian
    shadow belum aktif, paksa jalankan shadow correction + OCR ulang sebagai
    fallback. Ini menangani kasus bayangan yang tidak terdeteksi oleh
    has_uneven_illumination() (threshold edge case).

    Hasil terbaik dipilih berdasarkan skor kualitas OCR berbobot.

    Args:
        image_path (str): Path ke file gambar.
        lang       (str): Kode bahasa Tesseract. Default "ind" (Indonesia).
                          Gunakan "ind+eng" untuk dokumen campuran bahasa.

    Returns:
        tuple:
            best_text  (str)       : Teks OCR dengan skor tertinggi.
            all_texts  (list[str]) : Semua hasil OCR (untuk voting nama).
        Mengembalikan ("", []) jika preprocessing gagal.
    """
    variants = preprocess_image(image_path, use_deskew=use_deskew)
    if not variants:
        return "", []

    # Buat semua kombinasi task (label, img_v, psm) sekaligus
    tasks = [
        (label, img_v, psm)
        for label, img_v in variants
        for psm in [6, 11, 3]
    ]

    def _run_ocr(label, img_v, psm):
        config = f"--oem 3 --psm {psm} -l {lang}"
        try:
            text  = pytesseract.image_to_string(img_v, config=config)
            score = _ocr_quality_score(text)
            log.debug(f"  OCR [{label}/psm{psm}] skor={score}")
            return (score, label, psm, text)
        except Exception as exc:
            log.debug(f"  OCR error [{label}/psm{psm}]: {exc}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
        futures = {executor.submit(_run_ocr, lbl, img, psm): (lbl, psm)
                   for lbl, img, psm in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        return "", []

    results.sort(reverse=True)
    best_score = results[0][0]

    # ── v5.8.2: Fallback — paksa shadow correction jika OCR masih gagal ───
    # Dijalankan jika skor terbaik sangat rendah, baik shadow variant sudah
    # aktif atau belum. Jika shadow sudah aktif tapi skor tetap rendah,
    # berarti kernel/teknik pertama tidak efektif → coba kernel lebih besar
    # dan teknik berbeda.
    if best_score < 50:
        log.debug(
            f"  ShadowFallback: skor terbaik={best_score} (< 50) "
            f"→ coba shadow correction agresif"
        )
        # Ambil raw_gray dari variants yang sudah ada
        raw_gray_img = None
        for lbl, img_v in variants:
            if lbl == "raw_gray":
                raw_gray_img = img_v
                break
        if raw_gray_img is not None:
            shadow_results = []
            clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            clahe_hi  = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

            # Coba beberapa konfigurasi shadow removal (kernel+target)
            fb_configs = [
                (0.15, 220.0, "fb15"),   # kernel 15%, target 220
                (0.20, 230.0, "fb20"),   # kernel 20%, target 230
                (0.10, 200.0, "fb10"),   # kernel 10%, target 200 (default)
            ]
            for kpct, ntarget, tag in fb_configs:
                try:
                    gs = remove_shadow(raw_gray_img,
                                       kernel_pct=kpct, norm_target=ntarget)
                    # CLAHE standar
                    fb_clahe = clahe_obj.apply(gs)
                    # CLAHE agresif (clipLimit tinggi)
                    fb_clahe_hi = clahe_hi.apply(gs)
                    # Otsu
                    _, fb_otsu = cv2.threshold(
                        gs, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )
                    # equalizeHist
                    fb_eq = cv2.equalizeHist(gs)

                    fb_imgs = [
                        (f"sh_{tag}",       fb_clahe),
                        (f"sh_{tag}_hi",    fb_clahe_hi),
                        (f"sh_{tag}_otsu",  fb_otsu),
                        (f"sh_{tag}_eq",    fb_eq),
                    ]
                    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
                        futs = {
                            executor.submit(_run_ocr, lbl, im, psm): (lbl, psm)
                            for lbl, im in fb_imgs
                            for psm in [6, 11, 3]
                        }
                        for future in as_completed(futs):
                            res = future.result()
                            if res is not None:
                                shadow_results.append(res)
                except Exception as e:
                    log.debug(f"  ShadowFallback [{tag}] error: {e}")

            if shadow_results:
                results.extend(shadow_results)
                results.sort(reverse=True)
                new_best = results[0][0]
                log.debug(
                    f"  ShadowFallback: skor baru terbaik={new_best} "
                    f"(dari {len(shadow_results)} kombinasi tambahan)"
                )

    best_text = results[0][3]
    all_texts = [r[3] for r in results]
    return best_text, all_texts


def _ocr_quality_score(text: str) -> int:
    """
    Skor kualitas OCR berbobot v4 — memilih teks paling informatif.

    Masalah skor v3: clahe/PSM 6 menghasilkan banyak kata ≥3 huruf dari noise
    (nama-nama acak Indonesia) sehingga c1 meledak dan mengalahkan strategi yang
    benar-benar membaca nama/data valid.

    Perbaikan v4:
      1. Kata ≥4 huruf (cap 60)                         — turun dari ×3 tanpa cap
      2. Keyword dokumen                                 × 8  (naik dari ×5)
      3. Pola data terstruktur (tanggal, NIK)            × 6  (naik dari ×4)
      4. BONUS data valid (ditingkatkan):
         - Baris nama kapital murni (2-5 kata)           × 15 per baris
         - Baris nama prefixed "N. NAMA"                 × 12 per baris
         - Tanggal lengkap DD-MM/DD/YYYY                 × 10
         - Nama kota yang dikenal                        × 10
      5. Penalti GANDA lebih agresif:
         - Penalti baris noise (tidak ada kata ≥3 huruf) × 0.9
         - Penalti rasio token pendek/simbol             proporsional
    """
    KOTA_LIST = {
        "bekasi","jakarta","bandung","surabaya","medan","tangerang",
        "depok","bogor","semarang","yogyakarta","karawang","cikarang",
        "cimahi","sukabumi","tasikmalaya","cirebon","serang","cilegon",
        "solo","malang","denpasar","palembang","makassar","balikpapan",
        "samarinda","pontianak","banjarmasin","pekanbaru","mataram",
    }
    KW = [
        "nama","nik","alamat","lahir","pekerjaan","kewarganegaraan",
        "kawin","berlaku","kecamatan","kelurahan","kabupaten","provinsi",
        "kartu","keluarga","surat","ijin","izin","mengemudi","indonesia",
        "driving","licence","wanita","laki",
    ]

    # Komponen 1: kata ≥4 huruf (bobot kecil, diberi cap 60)
    c1 = min(len(re.findall(r"[A-Za-z]{4,}", text)), 60)

    # Komponen 2: keyword dokumen (bobot besar)
    c2 = sum(8 for k in KW if k in text.lower())

    # Komponen 3: pola data terstruktur
    c3 = (len(re.findall(r"\d{2}[-/]\d{2}[-/]\d{4}", text)) * 6 +
          len(re.findall(r"\b\d{14,16}\b", text)) * 6 +
          len(re.findall(r"\b\d{10,13}\b", text)) * 3)

    # Komponen 4: bonus data valid
    # 4a. Baris nama murni kapital (2-5 kata, tanpa digit)
    name_lines_clean = [
        ln.strip() for ln in text.split("\n")
        if re.match(r"^[A-Z][A-Z\s]{3,35}$", ln.strip())
        and 2 <= len(ln.strip().split()) <= 5
        and not re.search(r"\d", ln)
        and len(ln.strip()) >= 5
    ]
    # 4b. Baris nama dengan prefix nomor: "1. NAMA LENGKAP"
    name_lines_prefixed = [
        ln.strip() for ln in text.split("\n")
        if re.match(r"^\d+\.\s+[A-Z][A-Z\s]{3,35}$", ln.strip())
        and not re.search(r"\d", re.sub(r"^\d+\.\s+", "", ln.strip()))
    ]

    c4 = (len(name_lines_clean) * 15 +
          len(name_lines_prefixed) * 12 +
          len(re.findall(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}", text)) * 10 +
          sum(10 for kota in KOTA_LIST if kota in text.lower()))

    # Penalti 1: baris noise (tidak punya kata ≥3 huruf) — lebih agresif
    lines = [ln for ln in text.split("\n") if ln.strip()]
    noise_lines = sum(1 for ln in lines if not re.search(r"[A-Za-z]{3,}", ln))
    penalty1 = int((c1 + c2 + c3) * (noise_lines / max(1, len(lines))) * 0.9)

    # Penalti 2: rasio token pendek (<3 char alfanumerik) — deteksi karakter noise
    all_tokens = re.findall(r"\S+", text)
    short_tokens = sum(1 for t in all_tokens if len(re.sub(r"[^A-Za-z0-9]", "", t)) < 3)
    if all_tokens:
        short_ratio = short_tokens / len(all_tokens)
        # Jika >50% token adalah pendek/simbol, beri penalti ekstra
        penalty2 = int((c1 + c2 + c3 + c4) * max(0.0, short_ratio - 0.5) * 1.0)
    else:
        penalty2 = 0

    return c1 + c2 + c3 + c4 - penalty1 - penalty2

def clean_text(raw: str) -> str:
    """
    Membersihkan teks OCR dari noise karakter dan whitespace berlebih.

    Proses pembersihan:
      1. Hapus karakter non-printable di luar rentang ASCII + Latin Extended
      2. Normalisasi spasi/tab berturutan menjadi satu spasi
      3. Normalisasi baris kosong berturutan menjadi maksimal dua baris

    Args:
        raw (str): Teks mentah dari Tesseract.

    Returns:
        str: Teks yang sudah dibersihkan.
    """
    text = re.sub(r"[^\x20-\x7E\n\t\u00C0-\u024F]", " ", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
