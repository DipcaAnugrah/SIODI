"""
debug_demo.py — Mode Debug dan Mode Demo
=========================================
  - debug_single_file(): tampilkan semua hasil OCR + analisis lengkap (Modul 11)
  - create_demo_structure(): buat folder demo sintetik + ground_truth.csv (Modul 12)

Cara pakai:
  python main.py --debug "Dokumen/subfolder/KTP.jpeg"
  python main.py --demo

Perbaikan v5.8.2:
  - debug_single_file() sekarang menggunakan extract_text() dari preprocessing.py
    (bukan memanggil preprocess_image + OCR manual). Ini memastikan semua logika
    shadow fallback di extract_text() juga aktif saat debugging.
  - Tetap menampilkan detail per-variant melalui preprocess_image() terpisah
    untuk keperluan diagnostik visual.
"""
import os, re, csv, logging
import cv2
import pytesseract

from preprocessing import preprocess_image, extract_text, _ocr_quality_score, clean_text
from extractor import classify_document, extract_fields

log = logging.getLogger("otomatisasi_dokumen")

def debug_single_file(image_path: str, lang: str = "ind") -> None:
    """
    Tampilkan semua output OCR + hasil analisis lengkap untuk satu file.

    Perbaikan v5.8.2: menggunakan extract_text() untuk mendapatkan hasil OCR
    terbaik (termasuk shadow fallback), lalu juga menampilkan detail per-variant
    dari preprocess_image() untuk keperluan diagnostik.

    Args:
        image_path (str): Path ke file gambar yang akan di-debug.
        lang       (str): Kode bahasa Tesseract.
    """
    SEP = "=" * 65

    print(f"\n{SEP}")
    print(f"  DEBUG: {image_path}")
    print(SEP)

    if not os.path.exists(image_path):
        print(f"\n  [ERROR] File tidak ditemukan: {image_path}")
        print("  Pastikan path sudah benar dan file ada.")
        return

    # ── Bagian 1: Tampilkan detail per-variant (diagnostik) ──────────────
    variants = preprocess_image(image_path)
    if not variants:
        print("\n  [ERROR] Gagal membaca gambar.")
        print("  Pastikan file tidak korup dan formatnya didukung.")
        return

    all_results = []
    for label, img_v in variants:
        for psm in [6, 11, 3]:
            config = f"--oem 3 --psm {psm} -l {lang}"
            try:
                text  = pytesseract.image_to_string(img_v, config=config)
                score = _ocr_quality_score(text)
                raw_len = len(re.findall(r"[A-Za-z0-9]", text))
                all_results.append((score, label, psm, text, raw_len))
                print(f"\n  [Strategi: {label:14s} | PSM: {psm} | Karakter: {raw_len} | Skor: {score}]")
                snippet = text.strip()
                print(snippet[:400] if snippet else "  (kosong)")
            except Exception as e:
                print(f"  [ERROR OCR] [{label}/psm{psm}]: {e}")

    all_results.sort(reverse=True)

    # ── Bagian 2: Gunakan extract_text() untuk hasil terbaik ─────────────
    # extract_text() menjalankan SEMUA logika termasuk shadow fallback,
    # sehingga hasilnya bisa jauh lebih baik dari per-variant di atas.
    print(f"\n{'-' * 65}")
    print("  Menjalankan extract_text() (termasuk shadow fallback)...")
    print(f"{'-' * 65}")
    best_text, all_texts = extract_text(image_path, lang=lang)

    if best_text.strip():
        et_score = _ocr_quality_score(best_text)
        et_len   = len(re.findall(r"[A-Za-z0-9]", best_text))
        print(f"  extract_text() terbaik: skor={et_score}, {et_len} karakter")
        # Gabungkan ke all_results jika extract_text menghasilkan skor lebih baik
        if not all_results or et_score > all_results[0][0]:
            all_results.insert(0, (et_score, "extract_text", 0, best_text, et_len))
            print("  -> Hasil extract_text() LEBIH BAIK dari per-variant (shadow fallback membantu)")
        elif best_text not in [r[3] for r in all_results]:
            all_results.append((et_score, "extract_text", 0, best_text, et_len))
        # Tambahkan semua teks dari extract_text ke pool
        for i, t in enumerate(all_texts):
            s = _ocr_quality_score(t)
            all_results.append((s, f"et_{i}", 0, t, len(re.findall(r"[A-Za-z0-9]", t))))
    else:
        print("  extract_text() menghasilkan teks kosong.")

    print(f"\n{SEP}")
    print("  HASIL ANALISIS LENGKAP")
    print(SEP)

    if not all_results:
        print("  Tidak ada hasil OCR. Periksa instalasi Tesseract.")
        return

    all_results.sort(reverse=True)
    best    = all_results[0]
    clean   = clean_text(best[3])
    all_c   = [clean_text(r[3]) for r in all_results]

    doc_type = classify_document(clean)
    fields   = extract_fields(clean, doc_type, all_c)

    print(f"  Strategi terbaik  : {best[1]} / PSM {best[2]} (skor={best[0]}, {best[4]} karakter)")
    print(f"  Jenis dokumen     : {doc_type.upper()}")

    if doc_type == "kk":
        # Tampilkan 8 field KK
        print(f"  Nomor KK          : {fields.get('nomor_kk') or '-'}")
        print(f"  Kepala Keluarga   : {fields.get('nama_kepala') or '-'}")
        print(f"  Desa/Kelurahan    : {fields.get('desa_kelurahan') or '-'}")
        print(f"  Kecamatan         : {fields.get('kecamatan') or '-'}")
        print(f"  Kabupaten/Kota    : {fields.get('kabupaten_kota') or '-'}")
        print(f"  Provinsi          : {fields.get('provinsi') or '-'}")
        print(f"  RT/RW             : {fields.get('rtrw') or '-'}")
        alamat_disp = fields.get("alamat", "")
        if len(alamat_disp) > 80:
            alamat_disp = alamat_disp[:80] + "..."
        print(f"  Alamat            : {alamat_disp or '-'}")
    elif doc_type == "sim":
        print(f"  No. SIM           : {fields.get('no_sim') or '-'}")
        print(f"  Nama              : {fields['nama'] or '-'}")
        print(f"  Tempat Lahir      : {fields['tempat_lahir'] or '-'}")
        print(f"  Tanggal Lahir     : {fields['tanggal_lahir'] or '-'}")
        print(f"  Jenis Kelamin     : {fields['jenis_kelamin'] or '-'}")
        alamat_disp = fields.get('alamat', '')
        if len(alamat_disp) > 80:
            alamat_disp = alamat_disp[:80] + "..."
        print(f"  Alamat            : {alamat_disp or '-'}")
    else:
        print(f"  Nama              : {fields['nama'] or '-'}")
        print(f"  NIK               : {fields['nik'] or '-'}")
        print(f"  Tempat Lahir      : {fields['tempat_lahir'] or '-'}")
        print(f"  Tanggal Lahir     : {fields['tanggal_lahir'] or '-'}")
        print(f"  Jenis Kelamin     : {fields['jenis_kelamin'] or '-'}")
        alamat_disp = fields['alamat']
        if len(alamat_disp) > 80:
            alamat_disp = alamat_disp[:80] + "..."
        print(f"  Alamat            : {alamat_disp or '-'}")

    pct = fields['_field_completeness'] * 100
    print(
        f"  Field completeness: "
        f"{fields['_fields_filled']}/{fields['_fields_total']} ({pct:.0f}%)"
    )
    print(f"\n  Teks bersih (strategi terbaik):\n{'-'*40}")
    print(clean[:1200] if clean else "(kosong)")
    print(SEP + "\n")


# =============================================================================
# MODULE 12 — MODE DEMO
#
# Membuat folder dan gambar sintetik untuk pengujian tanpa data nyata.
# Berguna untuk memverifikasi instalasi dan memahami alur kerja sistem.
#
# Output yang dibuat:
#   Dokumen_Demo/
#     batch_A/
#       ktp_dipca.jpg      ← KTP dengan semua field
#       kk_sri.jpg         ← KK dengan Nama Kepala Keluarga
#     batch_B/
#       sim_budi.jpg       ← SIM
#       unknown_doc.jpg    ← dokumen tidak dikenal (akan masuk ERROR)
#     ground_truth.csv     ← contoh file ground truth untuk F1 evaluation
# =============================================================================
def create_demo_structure(base: str = "Dokumen_Demo") -> None:
    """
    Membuat folder demo dengan gambar sintetik dan ground truth.

    Menggunakan cv2.putText untuk menulis teks pada gambar putih.
    Berguna untuk memverifikasi instalasi tanpa perlu data nyata.

    Args:
        base (str): Nama folder demo. Default "Dokumen_Demo".
    """
    import numpy as np

    samples = [
        {
            "subfolder": "batch_A",
            "filename" : "ktp_dipca.jpg",
            "lines"    : [
                "PROVINSI JAWA BARAT",
                "KABUPATEN BEKASI",
                "NIK : 3216082501030001",
                "Nama : Dipca Anugrah",
                "Tempat/Tgl Lahir : BEKASI, 25-01-2003",
                "Jenis Kelamin : LAKI-LAKI",
                "Alamat : JL. MERDEKA NO. 1 RT 001/002",
                "Kecamatan : CIKARANG BARAT",
                "Berlaku Hingga : SEUMUR HIDUP",
            ],
        },
        {
            "subfolder": "batch_A",
            "filename" : "kk_sri.jpg",
            "lines"    : [
                "KARTU KELUARGA",
                "No KK : 5103060512010001",
                "Nama Kepala Keluarga : Sri Rejeki Kumalaputri",
                "Desa/Kelurahan : KEROBOKAN KAJA",
                "Kecamatan : KUTA UTARA",
                "Kabupaten/Kota : BADUNG",
                "Status Hubungan Dalam Keluarga",
                "Dinas Kependudukan Dan Pencatatan Sipil",
            ],
        },
        {
            "subfolder": "batch_B",
            "filename" : "sim_budi.jpg",
            "lines"    : [
                "SURAT IZIN MENGEMUDI",
                "POLRI",
                "Nama : Budi Santoso",
                "Tempat/Tgl Lahir : JAKARTA, 10-06-1990",
                "Jenis Kelamin : LAKI-LAKI",
                "Golongan : B1",
                "Berlaku Hingga : 17-08-2030",
            ],
        },
        {
            "subfolder": "batch_B",
            "filename" : "unknown_doc.jpg",
            "lines"    : [
                "SURAT KETERANGAN",
                "Dokumen tidak dikenal oleh sistem.",
                "File ini akan masuk ke folder ERROR/.",
            ],
        },
    ]

    print(f"\nMembuat folder demo: {base}/")
    for s in samples:
        folder = os.path.join(base, s["subfolder"])
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, s["filename"])

        img = np.ones((420, 740, 3), dtype="uint8") * 255
        y   = 38
        for line in s["lines"]:
            cv2.putText(
                img, line, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )
            y += 34
        cv2.imwrite(path, img)
        print(f"  Dibuat: {path}")

    # Buat file ground_truth.csv contoh
    gt_path = os.path.join(base, "ground_truth.csv")
    gt_rows = [
        {
            "nama_file"      : "ktp_dipca.jpg",
            "nama"           : "Dipca Anugrah",
            "nik"            : "3216082501030001",
            "tempat_lahir"   : "Bekasi",
            "tanggal_lahir"  : "25-01-2003",
            "jenis_kelamin"  : "LAKI-LAKI",
            "jenis_dokumen"  : "ktp",
            # Field KK kosong untuk KTP
            "nomor_kk"       : "",
            "nama_kepala"    : "",
            "desa_kelurahan" : "",
            "kecamatan"      : "",
            "kabupaten_kota" : "",
            "provinsi"       : "",
            "rtrw"           : "",
        },
        {
            "nama_file"      : "kk_sri.jpg",
            "nama"           : "Sri Rejeki Kumalaputri",
            "nik"            : "",
            "tempat_lahir"   : "",
            "tanggal_lahir"  : "",
            "jenis_kelamin"  : "",
            "jenis_dokumen"  : "kk",
            # Field KK lengkap
            "nomor_kk"       : "5103060512010001",
            "nama_kepala"    : "Sri Rejeki Kumalaputri",
            "desa_kelurahan" : "KEROBOKAN KAJA",
            "kecamatan"      : "KUTA UTARA",
            "kabupaten_kota" : "BADUNG",
            "provinsi"       : "BALI",
            "rtrw"           : "",
        },
        {
            "nama_file"      : "sim_budi.jpg",
            "nama"           : "Budi Santoso",
            "nik"            : "",
            "tempat_lahir"   : "Jakarta",
            "tanggal_lahir"  : "10-06-1990",
            "jenis_kelamin"  : "LAKI-LAKI",
            "jenis_dokumen"  : "sim",
            # Field KK kosong untuk SIM
            "nomor_kk"       : "",
            "nama_kepala"    : "",
            "desa_kelurahan" : "",
            "kecamatan"      : "",
            "kabupaten_kota" : "",
            "provinsi"       : "",
            "rtrw"           : "",
        },
    ]
    cols = [
        "nama_file", "nama", "nik", "tempat_lahir",
        "tanggal_lahir", "jenis_kelamin", "jenis_dokumen",
        "nomor_kk", "nama_kepala", "desa_kelurahan",
        "kecamatan", "kabupaten_kota", "provinsi", "rtrw",
    ]
    with open(gt_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(gt_rows)
    print(f"  Dibuat: {gt_path}  (contoh ground truth)")

    print(f"\nFolder demo siap. Cara menjalankan:\n")
    print(f"  # Tanpa evaluasi F1:")
    print(f'  python main.py --input "{base}"')
    print(f"\n  # Dengan evaluasi Precision/Recall/F1:")
    print(f'  python main.py --input "{base}" \\')
    print(f'      --ground-truth "{base}/ground_truth.csv"')
    print()
