# ============================================================
# build_exe.spec — PyInstaller spec file untuk SiODI GUI
# ============================================================
#
# CARA PENGGUNAAN:
#
# 1. Install PyInstaller:
#       pip install pyinstaller
#    atau:
#       python -m pip install pyinstaller
#
# 2. Verifikasi instalasi:
#       python -m PyInstaller --version
#    Jika muncul nomor versi (misalnya 6.x.x), berarti berhasil terpasang.
#
# 3. Build aplikasi dari folder project:
#       python -m PyInstaller --clean build_exe.spec
#    atau pada Windows:
#       py -m PyInstaller --clean build_exe.spec
#
# Catatan:
#   * Jangan gunakan `pyinstaller build_exe.spec` jika folder Scripts Python
#     belum masuk ke PATH.
#   * Perintah `python -m PyInstaller` lebih aman dan konsisten di semua
#     instalasi Python Windows.
#
# 4. Hasil .exe ada di:
#       dist/SiODI/SiODI.exe       <- folder (bawa semua isi folder)
#       dist/SiODI_portable.exe    <- single-file (lambat startup, lebih portable)
#
# CATATAN PENTING:
# ─────────────────
# • Tesseract OCR HARUS diinstal terpisah di komputer target
# • Atau sertakan tesseract.exe di dalam folder dist/ (lihat komentar di bawah)
# • Bahasa Indonesia (ind.traineddata) juga harus tersedia
#
# STRUKTUR DISTRIBUSI YANG DISARANKAN:
# ─────────────────────────────────────
# SiODI/
# ├── SiODI.exe              <- aplikasi utama
# ├── Tesseract-OCR/         <- (opsional) sertakan Tesseract portable
# │   ├── tesseract.exe
# │   └── tessdata/
# │       └── ind.traineddata
# ├── Dokumen/               <- folder input dokumen (buat manual)
# └── config.json            <- dibuat otomatis saat pertama jalan
# ============================================================

import sys
import os
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# ── Path project ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))
ASSETS_DIR  = os.path.join(PROJECT_DIR, "assets")

# ── Hidden imports (modul yang tidak terdeteksi otomatis oleh PyInstaller) ────
HIDDEN_IMPORTS = [
    # Library pihak ketiga
    "customtkinter",
    "PIL._tkinter_finder",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ImageFilter",
    "PIL.ImageEnhance",
    "cv2",
    "pytesseract",
    "numpy",
    "tqdm",
    # tkinter
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
    # Stdlib (biasanya sudah terdeteksi, tapi aman untuk eksplisit)
    "json",
    "threading",
    "queue",
    "logging",
    "webbrowser",
    "io",
    "pathlib",
    "datetime",
    "contextlib",
    "subprocess",
    # Modul lokal proyek
    "config",
    "pipeline",
    "preprocessing",
    "extractor",
    "file_manager",
    "evaluator",
    "logger",
    "debug_demo",
    "diag_tilted",
]

# ── Data tambahan yang harus disertakan ───────────────────────────────────────
# collect_data_files akan mengambil semua file non-Python dari package tsb.
DATAS = collect_data_files("customtkinter")

# Sertakan config.json bawaan
if os.path.isfile(os.path.join(PROJECT_DIR, "config.json")):
    DATAS += [(os.path.join(PROJECT_DIR, "config.json"), ".")]

# Tambahkan folder assets jika ada (ikon, gambar, dll.)
if os.path.isdir(ASSETS_DIR):
    DATAS.append((ASSETS_DIR, "assets"))

# ── Analisis dependensi ───────────────────────────────────────────────────────
a = Analysis(
    scripts       = [os.path.join(PROJECT_DIR, "app.py")],
    pathex        = [PROJECT_DIR],
    binaries      = [],
    datas         = DATAS,
    hiddenimports = HIDDEN_IMPORTS,
    hookspath     = [],
    hooksconfig   = {},
    runtime_hooks = [],
    excludes      = [
        # Hapus paket besar yang tidak diperlukan untuk mengurangi ukuran .exe
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "sphinx",
        "pytest",
        "setuptools",
        "pkg_resources",
    ],
    win_no_prefer_redirects = False,
    win_private_assemblies  = False,
    noarchive               = False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ── Mode FOLDER (disarankan — startup lebih cepat) ────────────────────────────
exe_folder = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries              = True,
    name                          = "SiODI",
    debug                         = False,
    bootloader_ignore_signals     = False,
    strip                         = False,
    upx                           = True,   # kompres dengan UPX jika tersedia
    console                       = False,  # False = tidak tampil terminal hitam
    disable_windowed_traceback    = False,
    target_arch                   = None,
    codesign_identity             = None,
    entitlements_file             = None,
    icon = (
        os.path.join(ASSETS_DIR, "icon.ico")
        if os.path.isfile(os.path.join(ASSETS_DIR, "icon.ico"))
        else None
    ),
)

coll = COLLECT(
    exe_folder,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip      = False,
    upx        = True,
    upx_exclude = [],
    name       = "SiODI",
)


# ── Mode SINGLE FILE (lebih lambat startup, ~80–150 MB) ───────────────────────
# Aktifkan blok ini dan komentari blok FOLDER di atas jika ingin 1 file .exe:
#
# exe_onefile = EXE(
#     pyz,
#     a.scripts,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     [],
#     name    = "SiODI_portable",
#     debug   = False,
#     strip   = False,
#     upx     = True,
#     console = False,
#     icon = (
#         os.path.join(ASSETS_DIR, "icon.ico")
#         if os.path.isfile(os.path.join(ASSETS_DIR, "icon.ico"))
#         else None
#     ),
# )
