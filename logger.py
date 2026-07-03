"""
logger.py — Sistem Logging Dual-Level
======================================
Terminal (INFO, format ringkas) + File (DEBUG, format lengkap).
File log disimpan di <folder_input>/logs/proses_<timestamp>.log
"""
import os, sys, logging
from datetime import datetime


def setup_logger(output_root: str) -> logging.Logger:
    """
    Inisialisasi logger dengan dua handler: terminal (INFO) dan file (DEBUG).

    Args:
        output_root (str): Folder root — subfolder logs/ dibuat di sini.

    Returns:
        logging.Logger: Instance logger yang sudah dikonfigurasi.
    """
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"proses_{timestamp}.log")

    logger = logging.getLogger("otomatisasi_dokumen")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # hindari duplicate handler jika dipanggil ulang

    # Terminal — ringkas, INFO ke atas
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    # Paksa UTF-8 agar karakter Unicode (█, ░, dll) tidak crash di terminal Windows
    if hasattr(sh.stream, 'reconfigure'):
        try:
            sh.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    # File — lengkap, DEBUG ke atas
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info(f"Log file: {log_file}")
    return logger
