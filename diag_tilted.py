"""
diag_tilted.py — Diagnostik mendalam untuk gambar miring yang tidak terdeteksi.
Jalankan dari folder gui/:  python diag_tilted.py "path/ke/gambar.jpg"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

image_path = sys.argv[1] if len(sys.argv) > 1 else "Data_Uji/ERROR/Data-18/IMG_20260606_231731.jpg"

print(f"\n{'='*65}")
print(f"DIAGNOSTIK: {image_path}")
print(f"{'='*65}\n")

# 1. Baca gambar
from PIL import Image, ImageOps
pil = Image.open(image_path)
pil_exif = ImageOps.exif_transpose(pil).convert("RGB")
img = cv2.cvtColor(np.array(pil_exif), cv2.COLOR_RGB2BGR)
h, w = img.shape[:2]
print(f"[1] Dimensi setelah EXIF transpose: {w}x{h} px")

# 2. Test quick_score 4 rotasi
from preprocessing import _quick_score
scores = {
    0:   _quick_score(img, "ind"),
    90:  _quick_score(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), "ind"),
    180: _quick_score(cv2.rotate(img, cv2.ROTATE_180), "ind"),
    270: _quick_score(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), "ind"),
}
print(f"\n[2] Skor OCR 4 orientasi:")
for deg, sc in scores.items():
    best_mark = " <-- terbaik" if sc == max(scores.values()) else ""
    print(f"    {deg:3d} deg: skor={sc}{best_mark}")

# 3. Test setiap Pass deskew
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
from preprocessing import (
    _estimate_skew_card_contour,
    _estimate_skew_minarearect,
    _estimate_skew_hough,
    _estimate_skew_projection,
)

print(f"\n[3] Estimasi sudut kemiringan setiap Pass:")
r0 = _estimate_skew_card_contour(gray)
print(f"    Pass 0 (CardContour) : {r0}")

r1 = _estimate_skew_minarearect(gray)
print(f"    Pass 1 (MinAreaRect) : {r1}")

r2 = _estimate_skew_hough(gray, w)
print(f"    Pass 2 (HoughLinesP): {r2}")

r3 = _estimate_skew_projection(gray)
print(f"    Pass 3 (Projection)  : {r3}")

# 4. Test deskew penuh
from preprocessing import deskew_image
img_deskew, total_skew = deskew_image(img)
print(f"\n[4] Total koreksi deskew_image(): {total_skew:.1f}°")

# 5. Test skor setelah deskew
gray_d = cv2.cvtColor(img_deskew, cv2.COLOR_BGR2GRAY)
r3_post = _estimate_skew_projection(gray_d)
print(f"\n[5] Projection setelah deskew: {r3_post}")

scores_post = {
    0:   _quick_score(img_deskew, "ind"),
    90:  _quick_score(cv2.rotate(img_deskew, cv2.ROTATE_90_CLOCKWISE), "ind"),
    180: _quick_score(cv2.rotate(img_deskew, cv2.ROTATE_180), "ind"),
    270: _quick_score(cv2.rotate(img_deskew, cv2.ROTATE_90_COUNTERCLOCKWISE), "ind"),
}
print(f"\n[6] Skor OCR 4 orientasi SETELAH deskew:")
for deg, sc in scores_post.items():
    best_mark = " <-- terbaik" if sc == max(scores_post.values()) else ""
    print(f"    {deg:3d} deg: skor={sc}{best_mark}")

# 6. Simpan gambar intermediate untuk inspeksi visual
out_dir = "diag_output"
os.makedirs(out_dir, exist_ok=True)

# Simpan gambar asli (setelah EXIF)
cv2.imwrite(f"{out_dir}/01_original.jpg", img)

# Simpan binary (seperti yang dipakai Projection)
scale = min(1.0, 400.0 / max(h, w))
small = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
_, binary_inv = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite(f"{out_dir}/02_binary_inv.jpg", binary_inv)

# Simpan Canny edges
canny = cv2.Canny(cv2.GaussianBlur(small, (5,5), 0), 30, 100)
cv2.imwrite(f"{out_dir}/03_canny.jpg", canny)

# Simpan gambar setelah deskew
cv2.imwrite(f"{out_dir}/04_deskewed.jpg", img_deskew)

# Simpan 4 rotasi setelah deskew
for deg, fn in [(90, "05_deskew_rot90"), (180, "06_deskew_rot180"), (270, "07_deskew_rot270")]:
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    cv2.imwrite(f"{out_dir}/{fn}.jpg", cv2.rotate(img_deskew, rot_map[deg]))

print(f"\n[7] Gambar diagnostik disimpan di: {out_dir}/")
print(f"    01_original.jpg    - gambar asli setelah EXIF")
print(f"    02_binary_inv.jpg  - binary yang dipakai Projection Profile")
print(f"    03_canny.jpg       - edge detection")
print(f"    04_deskewed.jpg    - hasil deskew akhir")
print(f"    05-07_...          - deskew + rotasi 90/180/270")
print(f"\n{'='*65}")
print(f"Periksa gambar di folder {out_dir}/ untuk melihat masalah!")
print(f"{'='*65}\n")
