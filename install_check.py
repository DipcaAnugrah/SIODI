#!/usr/bin/env python3
"""
install_check.py — Verifikasi Instalasi SiODI
==============================================
Jalankan script ini sebelum menggunakan aplikasi untuk memastikan
semua dependensi sudah terinstal dengan benar.

Cara pakai:
    python install_check.py

Dihasilkan oleh: Sistem OCR Dokumen Identitas GUI v5.7
"""
import sys, os

LINE = "=" * 60
OK   = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

def check(label: str, fn, critical: bool = True):
    try:
        result = fn()
        print(f"{OK} {label}" + (f": {result}" if result else ""))
        return True
    except Exception as e:
        prefix = FAIL if critical else WARN
        print(f"{prefix} {label}: {e}")
        return not critical

def main():
    print(f"\n{LINE}")
    print("  VERIFIKASI INSTALASI SiODI — Sistem OCR Dokumen Identitas")
    print(LINE)
    print()

    all_ok = True

    # ── Python version ────────────────────────────────────────────────────────
    print("[ Python ]")
    ver = sys.version_info
    if ver >= (3, 10):
        print(f"{OK} Python {ver.major}.{ver.minor}.{ver.micro}")
    else:
        print(f"{FAIL} Python {ver.major}.{ver.minor} — butuh minimal 3.10")
        all_ok = False
    print()

    # ── GUI Framework ─────────────────────────────────────────────────────────
    print("[ GUI Framework ]")
    all_ok &= check("customtkinter", lambda: __import__("customtkinter").__version__)
    all_ok &= check("tkinter (bawaan Python)", lambda: __import__("tkinter").TkVersion)
    print()

    # ── Computer Vision ───────────────────────────────────────────────────────
    print("[ Computer Vision & Image ]")
    all_ok &= check("opencv-python (cv2)", lambda: __import__("cv2").__version__)
    all_ok &= check("Pillow (PIL)", lambda: __import__("PIL").__version__)
    all_ok &= check("numpy", lambda: __import__("numpy").__version__)
    print()

    # ── OCR ──────────────────────────────────────────────────────────────────
    print("[ OCR Engine ]")
    all_ok &= check("pytesseract", lambda: __import__("pytesseract").__version__)

    # Cek Tesseract binary
    import pytesseract
    # Coba dari PATH dulu
    tesseract_ok = False
    try:
        ver_str = str(pytesseract.get_tesseract_version())
        print(f"{OK} Tesseract binary: v{ver_str}")
        tesseract_ok = True
    except Exception:
        # Coba path Windows default
        win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for wp in win_paths:
            if os.path.isfile(wp):
                pytesseract.pytesseract.tesseract_cmd = wp
                try:
                    ver_str = str(pytesseract.get_tesseract_version())
                    print(f"{OK} Tesseract binary: v{ver_str} (ditemukan di: {wp})")
                    print(f"{WARN} Atur path ini di halaman Pengaturan aplikasi!")
                    tesseract_ok = True
                    break
                except Exception:
                    pass
        if not tesseract_ok:
            print(f"{FAIL} Tesseract binary tidak ditemukan!")
            print("      Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("      Linux  : sudo apt install tesseract-ocr tesseract-ocr-ind")
            print("      macOS  : brew install tesseract tesseract-lang")
            all_ok = False

    # Cek bahasa Indonesia
    if tesseract_ok:
        try:
            langs = pytesseract.get_languages()
            if "ind" in langs:
                print(f"{OK} Bahasa Indonesia (ind) tersedia")
            else:
                print(f"{WARN} Bahasa Indonesia (ind) TIDAK tersedia!")
                print("      Tambahkan saat instalasi Tesseract atau unduh:")
                print("      https://github.com/tesseract-ocr/tessdata → ind.traineddata")
        except Exception as e:
            print(f"{WARN} Tidak bisa cek bahasa: {e}")
    print()

    # ── Modul lokal ───────────────────────────────────────────────────────────
    print("[ Modul Sistem SiODI ]")
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    local_modules = [
        ("config.py",        "config"),
        ("pipeline.py",      "pipeline"),
        ("preprocessing.py", "preprocessing"),
        ("extractor.py",     "extractor"),
        ("file_manager.py",  "file_manager"),
        ("evaluator.py",     "evaluator"),
        ("logger.py",        "logger"),
        ("debug_demo.py",    "debug_demo"),
        ("app.py",           None),       # cek file ada, tidak perlu import
    ]
    for fname, modname in local_modules:
        fpath = os.path.join(_here, fname)
        if os.path.isfile(fpath):
            if modname:
                try:
                    __import__(modname)
                    print(f"{OK} {fname}")
                except ImportError as e:
                    print(f"{WARN} {fname} (ada, tapi ada import error): {e}")
            else:
                print(f"{OK} {fname} (ada)")
        else:
            print(f"{FAIL} {fname} — FILE TIDAK DITEMUKAN!")
            all_ok = False
    print()

    # ── Utilitas opsional ─────────────────────────────────────────────────────
    print("[ Utilitas Opsional ]")
    check("tqdm (progress bar CLI)", lambda: __import__("tqdm").__version__,
          critical=False)
    check("pyinstaller (untuk build .exe)",
          lambda: __import__("PyInstaller").__version__, critical=False)
    print()

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    print(LINE)
    if all_ok:
        print("  ✅ SEMUA DEPENDENSI SIAP — Jalankan: python app.py")
    else:
        print("  ❌ ADA MASALAH — Perbaiki dulu sebelum menjalankan aplikasi")
        print()
        print("  Instal semua paket Python:")
        print("    pip install -r requirements_gui.txt")
    print(LINE + "\n")


if __name__ == "__main__":
    main()
