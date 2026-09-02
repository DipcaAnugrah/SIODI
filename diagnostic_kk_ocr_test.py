"""
diagnostic_kk_ocr.py -- Visualisasi Diagnostik OCR Nomor KK
=============================================================
Script ini menghasilkan serangkaian gambar diagnostik yang menunjukkan
"di balik layar" proses OCR pada area Nomor KK (kk_maya_sari.jpg):

  1. Gambar asli + bounding box area Nomor KK
  2. Hasil 4 strategi preprocessing (CLAHE, Otsu, Adaptive, Raw Gray)
     pada area Nomor KK yang di-crop
  3. Overlay perbandingan digit Ground Truth vs hasil OCR per strategi
  4. Heatmap kesalahan per posisi digit
  5. Visualisasi confusion: digit yang sering salah dibaca
  6. Zoom-in piksel digit bermasalah
  7. Tabel ringkasan statistik
  8. Diagram alur pipeline OCR

Output: Folder 'diagnostic_output/' di direktori yang sama
"""
print("Script loaded OK")
