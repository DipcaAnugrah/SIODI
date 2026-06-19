"""
main.py — Entry Point CLI
==========================
Titik masuk utama sistem otomatisasi penamaan dan pengarsipan dokumen identitas digital.

Cara menjalankan:
  python main.py --input "Dokumen"
  python main.py --input "Dokumen" --ground-truth ground_truth.csv
  python main.py --debug "Dokumen/subfolder/KTP.jpeg"
  python main.py --demo
  python main.py --input "Dokumen" --dry-run
  python main.py --input "Dokumen" --file-template "{nik}_{nama}"
  python main.py --template-help
  python main.py --create-config

Flag lengkap: --input --lang --export --ground-truth --debug --demo
              --file-template --folder-template --separator
              --template-help --dry-run --no-deskew
              --config --create-config --tesseract-path --version
"""
import os, sys, argparse
import pytesseract

from config import (
    VERSION, DEFAULT_FILE_TEMPLATE, DEFAULT_FOLDER_TEMPLATE, DEFAULT_SEPARATOR,
    CONFIG_DEFAULTS, validate_template, print_template_help,
    load_config, create_config_template,
)
from pipeline import process_folder
from debug_demo import debug_single_file, create_demo_structure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            f"Sistem Otomatisasi Penamaan dan Pengarsipan Dokumen Identitas Digital v{VERSION}\n"
            "OCR Multi-Strategi + NLP Rule-Based | KTP / KK / SIM\n\n"
            "Fitur: klasifikasi otomatis, ekstraksi field, metadata JSON/CSV,\n"
            "logging, evaluasi Precision/Recall/F1, efisiensi waktu,\n"
            "konsistensi penamaan."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh penggunaan:\n"
            "  python main.py --demo\n"
            "  python main.py --input Dokumen\n"
            "  python main.py --input Dokumen --ground-truth gt.csv\n"
            "  python main.py --debug Dokumen/sub/KTP.jpeg\n"
        ),
    )

    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {VERSION}",
        help="Tampilkan versi program dan keluar.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str, default="Dokumen",
        help="Folder utama berisi subfolder dokumen (default: Dokumen)",
    )
    parser.add_argument(
        "--lang", "-l",
        type=str, default="ind",
        help=(
            "Kode bahasa Tesseract (default: ind).\n"
            "Gunakan 'ind+eng' untuk dokumen campuran bahasa."
        ),
    )
    parser.add_argument(
        "--export",
        type=str, default="both",
        choices=["json", "csv", "both"],
        help="Format export metadata — json, csv, atau both (default: both)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str, default=None,
        metavar="CSV",
        help=(
            "Path ke file CSV ground truth untuk evaluasi F1-Score.\n"
            "Kolom: nama_file, nama, nik, tempat_lahir,\n"
            "       tanggal_lahir, jenis_kelamin, jenis_dokumen\n"
            "Jalankan --demo untuk melihat contoh formatnya."
        ),
    )
    parser.add_argument(
        "--debug",
        type=str, default=None,
        metavar="FILE",
        help=(
            "Debug satu file — tampilkan semua hasil OCR + analisis.\n"
            "Contoh: --debug \"Dokumen/subfolder/KTP.jpeg\""
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Buat folder demo sintetik + ground_truth.csv untuk pengujian cepat",
    )
    parser.add_argument(
        "--file-template",
        type=str, default=DEFAULT_FILE_TEMPLATE,
        metavar="FORMAT",
        help=(
            "Template nama file output tanpa ekstensi. "
            "(default: \"{jenis}_{nama}\")\n"
            "Variabel: {jenis},{JENIS},{nama},{NAMA},{nik},{nik6},\n"
            "          {tgl},{tgl_compact},{tempat},{jk},{tanggal},{timestamp}\n"
            "Contoh: \"--file-template {nik}_{nama}\" "
            "-> 3216082501030001_dipca_anugrah.jpeg\n"
            "Jalankan --template-help untuk panduan lengkap."
        ),
    )
    parser.add_argument(
        "--folder-template",
        type=str, default=DEFAULT_FOLDER_TEMPLATE,
        metavar="FORMAT",
        help=(
            "Template subfolder output. (default: \"{JENIS}/{jenis}_{nama}\")\n"
            "Gunakan / untuk subfolder bertingkat.\n"
            "Contoh: \"--folder-template {JENIS}/{tanggal}\" "
            "-> KTP/20260308/\n"
            "Jalankan --template-help untuk panduan lengkap."
        ),
    )
    parser.add_argument(
        "--separator",
        type=str, default=DEFAULT_SEPARATOR,
        metavar="CHAR",
        help=(
            "Pemisah kata dalam nama pemilik. (default: _)\n"
            "Contoh: --separator - -> ktp-dipca-anugrah.jpeg"
        ),
    )
    parser.add_argument(
        "--template-help",
        action="store_true",
        help="Tampilkan panduan lengkap template penamaan lalu keluar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Mode preview — analisis tanpa memindahkan file.\n"
            "Gunakan untuk verifikasi rencana rename sebelum proses sesungguhnya."
        ),
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Nonaktifkan koreksi kemiringan otomatis (lebih cepat, akurasi bisa turun).",
    )
    parser.add_argument(
        "--config",
        type=str, default=None,
        metavar="FILE",
        help=(
            "File konfigurasi JSON/YAML. Flag CLI mengoverride nilai config.\n"
            "Contoh: --config config.json\n"
            "Buat template: --create-config"
        ),
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Buat file config.json template lalu keluar.",
    )
    parser.add_argument(
        "--tesseract-path",
        type=str, default=None,
        help=(
            "Path manual ke tesseract.exe (terutama untuk Windows).\n"
            'Contoh: --tesseract-path "C:/Program Files/Tesseract-OCR/tesseract.exe"'
        ),
    )

    args = parser.parse_args()

    # ── Mode instan tanpa OCR ─────────────────────────────────────────────────
    if args.template_help:
        print_template_help()
        sys.exit(0)

    if args.create_config:
        create_config_template("config.json")
        sys.exit(0)

    # ── Muat config file, lalu CLI mengoverride ───────────────────────────────
    cfg = dict(CONFIG_DEFAULTS)
    if args.config:
        try:
            cfg = load_config(args.config)
            print(f"[OK] Config dimuat: {args.config}")
        except (FileNotFoundError, ValueError) as e:
            print(str(e))
            sys.exit(1)

    # Resolusi akhir: CLI > config file > default
    input_dir  = args.input        if args.input       != "Dokumen"              else cfg["input"]
    lang       = args.lang         if args.lang         != "ind"                  else cfg["lang"]
    export_fmt = args.export       if args.export       != "both"                 else cfg["export"]
    file_tpl   = args.file_template   if args.file_template   != DEFAULT_FILE_TEMPLATE   else cfg["file_template"]
    folder_tpl = args.folder_template if args.folder_template != DEFAULT_FOLDER_TEMPLATE else cfg["folder_template"]
    sep        = args.separator    if args.separator   != DEFAULT_SEPARATOR       else cfg["separator"]
    dry_run    = args.dry_run      or cfg.get("dry_run", False)
    use_deskew = not args.no_deskew and cfg.get("deskew", True)
    gt_path    = args.ground_truth or cfg.get("ground_truth")
    tess_path  = args.tesseract_path or cfg.get("tesseract_path")

    # ── Validasi template ─────────────────────────────────────────────────────
    for tpl_name, tpl_val in [
        ("--file-template",   file_tpl),
        ("--folder-template", folder_tpl),
    ]:
        ok, msg = validate_template(tpl_val)
        if not ok:
            print(f"\n[ERROR] Template tidak valid ({tpl_name}): {msg}")
            print("  Jalankan --template-help untuk daftar variabel yang tersedia.")
            sys.exit(1)

    # ── Setup Tesseract ───────────────────────────────────────────────────────
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path

    if not args.demo:
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            print(f"\n[ERROR] Tesseract tidak ditemukan: {e}")
            print("  Windows : https://github.com/UB-Mannheim/tesseract/wiki")
            print("             lalu --tesseract-path C:/Program Files/Tesseract-OCR/tesseract.exe")
            print("  Linux   : sudo apt install tesseract-ocr tesseract-ocr-ind")
            print("  macOS   : brew install tesseract")
            sys.exit(1)

    # ── Jalankan mode ─────────────────────────────────────────────────────────
    if args.debug:
        debug_single_file(args.debug, lang=lang)

    elif args.demo:
        create_demo_structure("Dokumen_Demo")

    else:
        if not os.path.isdir(input_dir):
            print(f"\n[ERROR] Folder tidak ditemukan: '{input_dir}'")
            print("  Gunakan --demo untuk folder uji coba.")
            sys.exit(1)

        process_folder(
            root_input        = input_dir,
            lang              = lang,
            export_fmt        = export_fmt,
            ground_truth_path = gt_path,
            file_template     = file_tpl,
            folder_template   = folder_tpl,
            sep               = sep,
            dry_run           = dry_run,
            use_deskew        = use_deskew,
        )
