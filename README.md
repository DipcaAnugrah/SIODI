# 🗂️ SiODI — Sistem OCR Dokumen Identitas
## Versi GUI Desktop v5.7

**SiODI** adalah aplikasi desktop modern untuk otomatisasi penamaan dan pengarsipan dokumen identitas digital menggunakan OCR multi-strategi dan NLP rule-based.

---

## 📋 Daftar Isi

1. [Fitur Utama](#fitur-utama)
2. [Persyaratan Sistem](#persyaratan-sistem)
3. [Instalasi Cepat](#instalasi-cepat)
4. [Cara Menjalankan](#cara-menjalankan)
5. [Panduan Penggunaan GUI](#panduan-penggunaan-gui)
6. [Build ke File .EXE](#build-ke-file-exe)
7. [Struktur Project](#struktur-project)
8. [Template Penamaan](#template-penamaan)
9. [Format Ground Truth CSV](#format-ground-truth-csv)
10. [Troubleshooting](#troubleshooting)

---

## ✨ Fitur Utama

| Fitur | Keterangan |
|-------|-----------|
| **OCR Multi-Strategi** | 12 kombinasi preprocessing (CLAHE, Otsu, Adaptive, Raw) × PSM (3/6/11) |
| **Klasifikasi Otomatis** | KTP, KK (Kartu Keluarga), SIM — berbasis scoring NLP rule-based |
| **Ekstraksi Field** | Nama, NIK, Tempat/Tgl Lahir, Jenis Kelamin, Alamat, No. KK, No. SIM |
| **Template Fleksibel** | Nama file & subfolder dapat dikustomisasi dengan variabel dinamis |
| **Evaluasi Ilmiah** | Precision, Recall, F1-Score per field vs. ground truth CSV |
| **Laporan HTML** | Laporan interaktif lengkap dengan tabel, grafik, dan statistik |
| **Export Metadata** | JSON dan CSV untuk setiap file yang diproses |
| **Mode Dry-Run** | Preview rencana pengarsipan tanpa memindahkan file |
| **GUI Modern** | CustomTkinter — tampilan profesional, support Light/Dark mode |
| **Multi-thread** | Proses berjalan di background, UI tetap responsif |
| **Build .EXE** | Dapat dikemas menjadi aplikasi Windows standalone |

---

## 🖥️ Persyaratan Sistem

### Wajib
- **Python** 3.10 atau lebih baru
- **Tesseract OCR** dengan bahasa Indonesia (`ind.traineddata`)

### Python Packages
```
customtkinter >= 5.2.2
opencv-python >= 4.8.0
Pillow        >= 10.0.0
pytesseract   >= 0.3.10
numpy         >= 1.24.0
tqdm          >= 4.66.0
```

### Instalasi Tesseract (wajib)

**Windows:**
1. Unduh dari: https://github.com/UB-Mannheim/tesseract/wiki
2. Saat instalasi, centang "Additional language data → Indonesian"
3. Path default: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ind
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

---

## ⚡ Instalasi Cepat

```bash
# 1. Clone atau ekstrak project
cd path/ke/folder/project

# 2. (Opsional) Buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# 3. Instal semua dependensi
pip install -r requirements_gui.txt

# 4. Jalankan aplikasi GUI
python app.py
```

---

## 🚀 Cara Menjalankan

### Mode GUI (Utama)
```bash
python app.py
```

### Mode CLI (tetap tersedia)
```bash
# Proses batch
python main.py --input "Dokumen"

# Dengan evaluasi F1
python main.py --input "Dokumen" --ground-truth ground_truth.csv

# Debug satu file
python main.py --debug "Dokumen/subfolder/KTP.jpeg"

# Buat demo
python main.py --demo

# Preview tanpa memindahkan file
python main.py --input "Dokumen" --dry-run
```

---

## 📖 Panduan Penggunaan GUI

### Halaman 1 — Proses Batch 📂

Halaman utama untuk memproses seluruh folder dokumen.

**Langkah penggunaan:**

1. **Pilih Folder Input** — klik tombol `📁 Browse` dan pilih folder yang berisi subfolder dokumen
2. **Atur Opsi OCR:**
   - *Bahasa Tesseract:* `ind` (Indonesia), `ind+eng` (campuran), `eng` (Inggris)
   - *Format Export:* `both` (JSON+CSV), `json`, `csv`
   - *Koreksi Kemiringan (Deskew):* centang untuk gambar yang miring
   - *Mode Dry-Run:* centang untuk preview tanpa memindahkan file
3. **Atur Template Penamaan** (opsional, lihat [Template Penamaan](#template-penamaan))
4. **Ground Truth CSV** (opsional, untuk evaluasi F1-Score)
5. Klik **▶ MULAI PROSES**
6. Pantau progress bar dan log output secara real-time
7. Setelah selesai, klik **🌐 Buka Laporan HTML** untuk melihat laporan lengkap

**Output yang dihasilkan:**
```
Dokumen/
├── KTP/
│   └── ktp_nama_pemilik/
│       └── ktp_nama_pemilik.jpg       ← file yang sudah diarsipkan
├── KK/
│   └── kk_nama_kepala/
│       └── kk_nama_kepala.jpg
├── SIM/
│   └── sim_nama_pemilik/
│       └── sim_nama_pemilik.jpg
├── ERROR/                              ← dokumen tidak berhasil diproses
├── metadata/
│   ├── metadata_YYYYMMDD_HHMMSS.json  ← metadata semua file
│   ├── metadata_YYYYMMDD_HHMMSS.csv
│   ├── laporan_YYYYMMDD_HHMMSS.html   ← laporan HTML interaktif
│   └── evaluasi_YYYYMMDD_HHMMSS.json  ← metrik evaluasi
└── logs/
    └── proses_YYYYMMDD_HHMMSS.log     ← log lengkap
```

---

### Halaman 2 — Debug File Tunggal 🔍

Analisis mendalam satu file dokumen untuk keperluan troubleshooting.

**Menampilkan:**
- Semua 12 kombinasi strategi OCR dan skornya
- Strategi terbaik yang dipilih sistem
- Jenis dokumen yang terdeteksi
- Semua field yang berhasil diekstrak
- Field completeness score
- Teks bersih hasil OCR terbaik

**Gunakan ini ketika:**
- Dokumen masuk ke folder ERROR/ tanpa alasan jelas
- Field tertentu tidak terekstrak dengan benar
- Ingin mengetahui kualitas OCR pada gambar tertentu

---

### Halaman 3 — Buat Demo 🧪

Membuat folder demo sintetik untuk pengujian tanpa data dokumen nyata.

**Dokumen yang dibuat:**
- `ktp_dipca.jpg` — KTP dengan semua field lengkap
- `kk_sri.jpg` — Kartu Keluarga
- `sim_budi.jpg` — SIM (Surat Izin Mengemudi)
- `unknown_doc.jpg` — Dokumen tidak dikenal (akan masuk ERROR/)
- `ground_truth.csv` — Data evaluasi F1-Score contoh

---

### Halaman 4 — Pengaturan ⚙️

Konfigurasi global yang disimpan di `gui_config.json`.

| Pengaturan | Keterangan |
|-----------|-----------|
| **Tesseract Path** | Path ke `tesseract.exe` (wajib di Windows jika tidak di PATH) |
| **Bahasa Default** | Bahasa OCR default untuk semua sesi |
| **Export Default** | Format metadata default |
| **Template Default** | Template penamaan file dan folder default |
| **Pemisah Kata** | Karakter pemisah dalam nama slug (`_`, `-`, `.`) |
| **Deskew Default** | Aktifkan/nonaktifkan deskew secara default |
| **Tema** | Light / Dark / System |

**Tombol "✔ Cek Tesseract"** — verifikasi apakah Tesseract terdeteksi dengan benar sebelum memproses.

---

### Halaman 5 — Log Viewer 📋

Melihat isi file log dari sesi sebelumnya.

1. Pilih folder yang berisi subfolder `logs/` (biasanya folder input Anda)
2. Klik **📂 Muat File Log**
3. Pilih file log dari dropdown (diurutkan terbaru lebih dulu)

---

## 📦 Build ke File .EXE

### Persiapan

```bash
pip install pyinstaller
```

### Buat folder assets (opsional, untuk ikon)
```
project/
└── assets/
    └── icon.ico    ← ikon aplikasi (256×256 px, format .ico)
```

### Build

```bash
# Build folder (disarankan)
pyinstaller build_exe.spec

# Hasilnya ada di:
# dist/SiODI/SiODI.exe
```

### Distribusi

Kirimkan seluruh folder `dist/SiODI/` ke komputer target. Pastikan:
1. **Tesseract OCR** sudah terinstal di komputer target, ATAU
2. Sertakan folder `Tesseract-OCR/` di dalam `dist/SiODI/` (portable)
3. Atur path Tesseract di halaman **Pengaturan** saat pertama kali membuka aplikasi

**Catatan:** Jika Tesseract belum diatur, aplikasi tetap bisa dibuka — pengguna cukup mengisi path di halaman Pengaturan.

---

## 📁 Struktur Project

```
project/
│
├── app.py                  ← 🆕 Entry point GUI (CustomTkinter)
├── main.py                 ← Entry point CLI (argparse) — tetap berfungsi
│
├── pipeline.py             ← ✏️ Pipeline batch (dimodifikasi: +progress_callback)
├── config.py               ← Konfigurasi, konstanta, template
├── preprocessing.py        ← Preprocessing citra + OCR 12 kombinasi
├── extractor.py            ← Klasifikasi + ekstraksi field NLP
├── file_manager.py         ← Manajemen file & path output
├── evaluator.py            ← Evaluasi F1, HTML report, export metadata
├── logger.py               ← Dual logging terminal + file
├── debug_demo.py           ← Debug file tunggal + buat demo sintetik
│
├── config.json             ← Konfigurasi sistem (auto-generated)
├── gui_config.json         ← 🆕 Konfigurasi GUI (auto-generated)
│
├── requirements_gui.txt    ← 🆕 Dependensi lengkap
├── build_exe.spec          ← 🆕 PyInstaller spec untuk build .exe
├── README_GUI.md           ← 🆕 Dokumentasi ini
│
├── assets/                 ← 🆕 (opsional) Aset GUI
│   └── icon.ico
│
└── data_img/               ← Gambar contoh (tidak perlu untuk produksi)
    ├── Ktp.jpeg
    ├── KK.jpeg
    └── SIM_dipca_1.jpeg
```

**Keterangan:**
- 🆕 = File baru yang ditambahkan untuk GUI
- ✏️ = File lama yang dimodifikasi (backward-compatible)
- File lainnya = tidak diubah sama sekali

---

## 🏷️ Template Penamaan

Template menggunakan variabel dalam kurung kurawal `{variabel}`.

### Variabel Umum (KTP, KK, SIM)

| Variabel | Contoh Output | Keterangan |
|----------|--------------|-----------|
| `{jenis}` | `ktp` | Jenis dokumen (huruf kecil) |
| `{JENIS}` | `KTP` | Jenis dokumen (huruf besar) |
| `{nama}` | `dipca_anugrah` | Nama pemilik (slug, huruf kecil) |
| `{NAMA}` | `DIPCA_ANUGRAH` | Nama pemilik (huruf besar) |
| `{nik}` | `3216082501030001` | NIK 16 digit |
| `{nik6}` | `321608` | 6 digit pertama NIK |
| `{tgl}` | `25-01-2003` | Tanggal lahir (DD-MM-YYYY) |
| `{tgl_compact}` | `25012003` | Tanggal lahir tanpa pemisah |
| `{tempat}` | `bekasi` | Tempat lahir (slug) |
| `{jk}` | `L` atau `P` | Jenis kelamin (L/P) |
| `{tanggal}` | `20260308` | Tanggal proses (YYYYMMDD) |
| `{timestamp}` | `20260308_143022` | Timestamp proses |

### Variabel Khusus KK

| Variabel | Contoh Output |
|----------|--------------|
| `{nokk}` | `5103060512010001` |
| `{nokk6}` | `510306` |
| `{kepala}` | `sri_rejeki` |
| `{desa}` | `kerobokan_kaja` |
| `{kecamatan}` | `kuta_utara` |
| `{kabupaten}` | `badung` |
| `{provinsi}` | `bali` |

### Contoh Template

```
# Default (bawaan)
File:   {jenis}_{nama}           → ktp_dipca_anugrah.jpg
Folder: {JENIS}/{jenis}_{nama}   → KTP/ktp_dipca_anugrah/

# Dengan NIK di depan
File:   {nik}_{nama}             → 3216082501030001_dipca_anugrah.jpg

# Berdasarkan tanggal proses
Folder: {JENIS}/{tanggal}        → KTP/20260308/

# KK berdasarkan wilayah
Folder: {JENIS}/{kabupaten}/{kecamatan} → KK/badung/kuta_utara/
```

---

## 📊 Format Ground Truth CSV

File CSV untuk evaluasi Precision/Recall/F1-Score. Kolom yang diperlukan:

```csv
nama_file,nama,nik,tempat_lahir,tanggal_lahir,jenis_kelamin,jenis_dokumen,nomor_kk,nama_kepala,desa_kelurahan,kecamatan,kabupaten_kota,provinsi,rtrw
ktp_dipca.jpg,Dipca Anugrah,3216082501030001,Bekasi,25-01-2003,LAKI-LAKI,ktp,,,,,,
kk_sri.jpg,Sri Rejeki,,,,,kk,5103060512010001,Sri Rejeki,Kerobokan Kaja,Kuta Utara,Badung,Bali,
sim_budi.jpg,Budi Santoso,,Jakarta,10-06-1990,LAKI-LAKI,sim,,,,,,
```

**Tips:**
- Kolom yang tidak relevan untuk jenis dokumen tertentu → biarkan kosong
- Gunakan Mode Demo untuk melihat contoh format lengkap
- File ini dihasilkan otomatis saat Mode Demo dijalankan

---

## 🔧 Troubleshooting

### ❌ `customtkinter` tidak ditemukan
```bash
pip install customtkinter
```

### ❌ Tesseract tidak ditemukan
1. Pastikan Tesseract sudah terinstal (lihat [Instalasi Tesseract](#instalasi-tesseract-wajib))
2. Buka halaman **Pengaturan** → isi path `tesseract.exe`
3. Klik **✔ Cek Tesseract** untuk verifikasi

### ❌ Bahasa Indonesia tidak tersedia
Saat instalasi Tesseract Windows, pastikan centang:
`Additional language data → Indonesian`

Atau unduh manual `ind.traineddata` dari:
https://github.com/tesseract-ocr/tessdata

Dan letakkan di folder `tessdata/` Tesseract.

### ❌ Dokumen selalu masuk ERROR/
Gunakan halaman **🔍 Debug File** untuk melihat output OCR mentah.
Kemungkinan penyebab:
- Gambar terlalu gelap, buram, atau beresolusi rendah (minimal 800px)
- Tidak ada keyword pengenal dokumen yang terbaca
- Bahasa OCR tidak sesuai (coba `ind+eng`)

### ❌ Field tidak terekstrak (nama/NIK kosong)
- Pastikan label seperti "NIK :", "Nama :" terbaca di output debug
- Coba tanpa deskew jika gambar sudah lurus
- Periksa kecerahan dan kontras gambar

### ❌ .EXE gagal dijalankan
- Pastikan Visual C++ Redistributable terinstal di komputer target
- Jalankan sebagai Administrator jika ada masalah akses folder
- Periksa `gui_config.json` — hapus dan biarkan dibuat ulang jika korup

### ❌ GUI tidak muncul / crash saat startup
```bash
# Jalankan dari terminal untuk melihat error:
python app.py
```

---

## 📚 Referensi Teknis

- **OCR Engine:** Tesseract 5.x dengan LSTM neural network
- **Preprocessing:** CLAHE equalization, Otsu thresholding, Adaptive thresholding
- **Deskew:** Hough Line Transform untuk deteksi dan koreksi sudut kemiringan
- **Klasifikasi:** Scoring berbasis regex dengan threshold per jenis dokumen
- **Ekstraksi:** NLP rule-based 4-strategi berlapis dengan voting konsensus
- **Evaluasi:** Precision/Recall/F1 per field, field completeness score, efisiensi waktu
- **GUI:** CustomTkinter (wrapper modern di atas Tkinter bawaan Python)
- **Threading:** `threading.Thread` + `queue.Queue` untuk UI non-blocking

---

*SiODI v5.7 — Sistem Otomatisasi Penamaan dan Pengarsipan Dokumen Identitas Digital*
*Dikembangkan dengan Python · CustomTkinter · Tesseract OCR · OpenCV*
