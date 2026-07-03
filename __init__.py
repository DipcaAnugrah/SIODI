"""
Sistem Otomatisasi Penamaan dan Pengarsipan Dokumen Identitas Digital
Berbasis OCR Multi-Strategi dan NLP Rule-Based  (v5.7)

Package Structure:
    config.py       — Konstanta, template penamaan, konfigurasi JSON/YAML
    logger.py       — Logging dual-level (terminal INFO + file DEBUG)
    preprocessing.py — Orientasi, deskewing, 4x preprocessing, OCR 12 kombinasi
    extractor.py    — Klasifikasi KTP/KK/SIM, ekstraksi field, voting konsensus
    file_manager.py — Build path output, pindah file, anti-duplikat
    evaluator.py    — Ekspor JSON/CSV, F1-Score, efisiensi waktu, laporan HTML
    pipeline.py     — Batch processing end-to-end
    debug_demo.py   — Mode debug satu file + mode demo sintetik
    main.py         — Entry point CLI (argparse)

Cara menjalankan:
    python main.py --input "Dokumen"
    python main.py --demo
    python main.py --debug "Dokumen/sub/KTP.jpeg"
"""

__version__ = "5.8"
__author__  = "Sistem OCR NLP Rule-Based"
