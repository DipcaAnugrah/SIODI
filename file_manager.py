"""
file_manager.py — Manajemen File dan Struktur Folder Output
============================================================
Menangani:
  - Penentuan path tujuan berdasarkan template (Modul 7)
  - Pemindahan file secara aman (copy2 + remove)
  - Penanganan duplikat nama (sufiks _1, _2, ...)
  - Struktur output: KTP/ KK/ SIM/ ERROR/ metadata/ logs/
"""
import os, re, shutil, logging

from config import DEFAULT_FILE_TEMPLATE, DEFAULT_FOLDER_TEMPLATE, DEFAULT_SEPARATOR
from config import build_template_context, apply_template

log = logging.getLogger("otomatisasi_dokumen")

def build_output_path(
    src_path        : str,
    root_input      : str,
    doc_type        : str,
    name            : str,
    success         : bool,
    fields          : dict = None,
    file_template   : str  = DEFAULT_FILE_TEMPLATE,
    folder_template : str  = DEFAULT_FOLDER_TEMPLATE,
    sep             : str  = DEFAULT_SEPARATOR,
) -> str:
    """
    Menentukan path tujuan file berdasarkan template penamaan yang dipilih user.

    Untuk dokumen VALID:
      - Nama file  ditentukan oleh file_template   (default: "{jenis}_{nama}")
      - Subfolder  ditentukan oleh folder_template (default: "{JENIS}/{jenis}_{nama}")
      - Contoh default : root/KTP/ktp_dipca_anugrah/ktp_dipca_anugrah.jpeg
      - Contoh kustom  : root/KTP/20260308/3216082501030001_dipca_anugrah.jpeg

    Untuk dokumen ERROR:
      Folder: <root_input>/ERROR/<subfolder_asal>/ (struktur asli dipertahankan)

    Jika file tujuan sudah ada (duplikat), sufiks _1, _2, dst. ditambahkan otomatis.

    Args:
        src_path        (str) : Path file sumber.
        root_input      (str) : Folder root pemrosesan.
        doc_type        (str) : 'ktp', 'kk', 'sim', atau 'unknown'.
        name            (str) : Slug nama pemilik (dari extract_fields).
        success         (bool): True jika klasifikasi dan nama berhasil.
        fields          (dict): Semua field hasil ekstraksi (untuk template kustom).
        file_template   (str) : Template nama file tanpa ekstensi.
        folder_template (str) : Template path subfolder. Gunakan "/" untuk bertingkat.
        sep             (str) : Pemisah kata dalam nama slug.

    Returns:
        str: Path tujuan absolut yang sudah siap digunakan.
    """
    ext                = os.path.splitext(src_path)[1].lower()
    original_subfolder = os.path.relpath(
        os.path.dirname(src_path), root_input
    )

    if success:
        ctx          = build_template_context(doc_type, fields or {"nama": name}, sep)
        file_part    = apply_template(file_template,   ctx, sep)
        folder_part  = apply_template(folder_template, ctx, sep)
        new_filename = file_part + ext
        folder_segs  = folder_part.split("/")
        dest_dir     = os.path.join(root_input, *folder_segs)
    else:
        new_filename = os.path.basename(src_path)
        dest_dir     = os.path.join(root_input, "ERROR", original_subfolder)

    os.makedirs(dest_dir, exist_ok=True)

    # Hindari overwrite file dengan nama sama
    dest_path = os.path.join(dest_dir, new_filename)
    counter   = 1
    while os.path.exists(dest_path):
        base      = os.path.splitext(new_filename)[0]
        dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter  += 1

    return dest_path


def move_file(src_path: str, dest_path: str) -> None:
    """
    Salin file ke tujuan lalu hapus sumber (safe atomic move).

    Menggunakan copy2 + remove (bukan shutil.move) agar metadata file
    seperti tanggal modifikasi ikut tersalin.

    Args:
        src_path  (str): Path file sumber.
        dest_path (str): Path file tujuan (sudah dibuat oleh build_output_path).
    """
    shutil.copy2(src_path, dest_path)
    os.remove(src_path)

