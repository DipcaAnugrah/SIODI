#!/usr/bin/env python3
"""
app.py — Antarmuka GUI Desktop: Sistem OCR Dokumen Identitas
=============================================================
Entry point GUI menggunakan CustomTkinter.
Menggantikan main.py CLI dengan antarmuka visual modern dan profesional.

Cara menjalankan:
    python app.py

Build ke .exe Windows:
    pyinstaller build_exe.spec

Fitur GUI:
  • Proses Batch  — proses folder penuh dengan progress bar real-time
  • Debug File    — analisis mendalam satu file dokumen
  • Buat Demo     — buat folder demo untuk pengujian
  • Pengaturan    — konfigurasi Tesseract, template, separator, tema
  • Log Viewer    — tampilkan log sesi saat ini & riwayat

Sistem: SiODI v5.7 | Python + CustomTkinter | KTP / KK / SIM
"""

import os, sys, json, threading, queue, logging, subprocess, webbrowser, io
from datetime import datetime
from contextlib import redirect_stdout
from pathlib import Path

# ── Pastikan folder script ada di sys.path ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# ═══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA & TEMA
# ═══════════════════════════════════════════════════════════════════════════════
APP_NAME     = "SiODI"
APP_SUBTITLE = "Sistem OCR Dokumen Identitas"
APP_VERSION  = "5.7"
WINDOW_W, WINDOW_H = 1280, 780

# Warna brand — Tema Terang Modern
C = {
    "navy":        "#111827",   # Hitam/navy gelap untuk teks & elemen dominan
    "navy_mid":    "#374151",   # Abu gelap untuk secondary
    "navy_light":  "#22C55E",   # Hijau aksen utama
    "accent":      "#22C55E",   # Hijau terang
    "accent_dark": "#16A34A",   # Hijau gelap hover
    "success":     "#22C55E",   # Hijau valid
    "warning":     "#F97316",   # Oranye untuk warning/running
    "error":       "#EF4444",   # Merah error
    "sidebar":     "#FFFFFF",   # Sidebar putih bersih
    "sidebar_btn": "#F3F4F6",   # Tombol sidebar abu terang
    "sidebar_act": "#ECFDF5",   # Sidebar active hijau muda
    "white":       "#FFFFFF",
    "gray_light":  "#F9FAFB",   # Background utama abu sangat terang
    "gray_mid":    "#E5E7EB",   # Garis pemisah
    "gray_text":   "#6B7280",   # Teks sekunder
    "text_dark":   "#111827",   # Teks utama gelap
}

# Jenis dokumen yang didukung
SUPPORTED_IMGS = [
    ("Gambar", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
    ("Semua file", "*.*"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM LOGGING HANDLER → antri ke Queue (thread-safe untuk GUI)
# ═══════════════════════════════════════════════════════════════════════════════
class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
class Sidebar(ctk.CTkFrame):
    """Panel navigasi kiri dengan tombol halaman dan info versi."""

    NAV_ITEMS = [
        ("batch",    "📂  Proses Batch"),
        ("debug",    "🔍  Debug File"),
        ("demo",     "🧪  Buat Demo"),
        ("settings", "⚙️   Pengaturan"),
        ("log",      "📋  Log Viewer"),
        ("about",    "ℹ️   Tentang"),
    ]

    def __init__(self, master, on_navigate):
        super().__init__(master, width=220, corner_radius=0,
                         fg_color=C["sidebar"])
        self.grid_propagate(False)
        self.on_navigate = on_navigate
        self._buttons: dict[str, ctk.CTkButton] = {}
        self._active = None
        self._build()

    def _build(self):
        self.grid_rowconfigure(len(self.NAV_ITEMS) + 2, weight=1)

        # Logo area
        logo_frame = ctk.CTkFrame(self, fg_color="transparent", height=80)
        logo_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(20, 4))
        logo_frame.grid_propagate(False)

        ctk.CTkLabel(logo_frame, text="🗂️", font=("Segoe UI Emoji", 32),
                     text_color=C["accent"]).pack(anchor="w")
        ctk.CTkLabel(logo_frame, text=APP_NAME,
                     font=ctk.CTkFont("Segoe UI", 22, "bold"),
                     text_color=C["navy"]).pack(anchor="w")

        # Garis pemisah
        ctk.CTkFrame(self, height=1, fg_color=C["gray_mid"]).grid(
            row=1, column=0, sticky="ew", padx=12, pady=8)

        # Tombol navigasi
        for i, (page_id, label) in enumerate(self.NAV_ITEMS):
            btn = ctk.CTkButton(
                self, text=label, anchor="w",
                font=ctk.CTkFont("Segoe UI", 13),
                height=42, corner_radius=8,
                fg_color="transparent",
                hover_color=C["sidebar_btn"],
                text_color=C["navy_mid"],
                command=lambda p=page_id: self.on_navigate(p),
            )
            btn.grid(row=i + 2, column=0, sticky="ew", padx=10, pady=2)
            self._buttons[page_id] = btn

        # Versi di bawah
        ctk.CTkLabel(self, text=f"v{APP_VERSION}",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=C["gray_text"]).grid(
            row=len(self.NAV_ITEMS) + 3, column=0, pady=(0, 16))

    def set_active(self, page_id: str):
        if self._active and self._active in self._buttons:
            self._buttons[self._active].configure(
                fg_color="transparent", text_color=C["navy_mid"])
        self._active = page_id
        if page_id in self._buttons:
            self._buttons[page_id].configure(
                fg_color=C["sidebar_act"], text_color=C["accent_dark"])


# ═══════════════════════════════════════════════════════════════════════════════
#  WIDGET HELPER: Card, Field Row, Section Label
# ═══════════════════════════════════════════════════════════════════════════════
def make_card(parent, **kwargs) -> ctk.CTkFrame:
    defaults = dict(corner_radius=12, fg_color=("white", "#2A2A2A"),
                    border_width=1, border_color=("#E5E7EB", "#3A3A3A"))
    defaults.update(kwargs)
    return ctk.CTkFrame(parent, **defaults)


def section_label(parent, text: str, row: int, col: int = 0, colspan: int = 1):
    ctk.CTkLabel(parent, text=text,
                 font=ctk.CTkFont("Segoe UI", 11, "bold"),
                 text_color=C["gray_text"]).grid(
        row=row, column=col, columnspan=colspan,
        sticky="w", padx=16, pady=(12, 2))


def make_browse_row(parent, label: str, var: tk.StringVar,
                    row: int, file_mode=True,
                    filetypes=None, title="Pilih"):
    """Baris label + entry + tombol browse."""
    ctk.CTkLabel(parent, text=label,
                 font=ctk.CTkFont("Segoe UI", 13)).grid(
        row=row, column=0, sticky="w", padx=(16, 8), pady=6)
    entry = ctk.CTkEntry(parent, textvariable=var,
                         font=ctk.CTkFont("Segoe UI", 12), height=34)
    entry.grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=6)

    def browse():
        if file_mode:
            path = filedialog.askopenfilename(
                title=title, filetypes=filetypes or SUPPORTED_IMGS)
        else:
            path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    ctk.CTkButton(parent, text="📁 Browse", width=100, height=34,
                  font=ctk.CTkFont("Segoe UI", 12),
                  command=browse).grid(row=row, column=2, padx=(0, 16), pady=6)
    return entry


# ═══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 1: PROSES BATCH
# ═══════════════════════════════════════════════════════════════════════════════
class BatchPage(ctk.CTkFrame):
    """Halaman utama: konfigurasi + jalankan batch OCR."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Variabel form
        self.v_input    = tk.StringVar()
        self.v_lang     = tk.StringVar(value=app.settings.get("default_lang", "ind"))
        self.v_export   = tk.StringVar(value=app.settings.get("default_export", "both"))
        self.v_file_tpl = tk.StringVar(value=app.settings.get("default_file_template",
                                                               "{jenis}_{nama}"))
        self.v_fol_tpl  = tk.StringVar(value=app.settings.get("default_folder_template",
                                                               "{JENIS}/{jenis}_{nama}"))
        self.v_sep      = tk.StringVar(value=app.settings.get("default_separator", "_"))
        self.v_gt       = tk.StringVar()
        self.v_deskew   = tk.BooleanVar(value=app.settings.get("use_deskew", True))
        self.v_dryrun   = tk.BooleanVar(value=False)
        self._html_report_path = None
        self._log_lines = 0

        self._build_header()
        self._build_config_area()
        self._build_progress_area()
        self._build_log_area()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, corner_radius=0, height=64,
                           fg_color=C["white"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(hdr, text="📂  Proses Batch Dokumen Identitas",
                     font=ctk.CTkFont("Segoe UI", 17, "bold"),
                     text_color=C["navy"]).grid(row=0, column=0, padx=20, pady=18, sticky="w")
        ctk.CTkLabel(hdr, text="KTP · KK · SIM",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color=C["accent"]).grid(row=0, column=1, padx=20, sticky="e")

    # ── Konfigurasi ───────────────────────────────────────────────────────────
    def _build_config_area(self):
        cfg = ctk.CTkScrollableFrame(self, corner_radius=0,
                                     fg_color=C["gray_light"],
                                     label_text="", height=310)
        cfg.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        cfg.grid_columnconfigure(0, weight=1)
        self._cfg_frame = cfg

        # ── Baris folder input ────────────────────────────────────────────────
        top = make_card(cfg)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        top.grid_columnconfigure(1, weight=1)
        section_label(top, "📁  FOLDER DOKUMEN", 0, colspan=3)
        make_browse_row(top, "Folder Input:", self.v_input, 1,
                        file_mode=False, title="Pilih Folder Dokumen Input")
        section_label(top, "📄  GROUND TRUTH (Opsional — untuk evaluasi F1-Score)", 2, colspan=3)
        make_browse_row(top, "File CSV:", self.v_gt, 3,
                        file_mode=True,
                        filetypes=[("CSV", "*.csv"), ("Semua", "*.*")],
                        title="Pilih File Ground Truth")

        # ── Row 2: Opsi OCR + Template ────────────────────────────────────────
        mid = ctk.CTkFrame(cfg, fg_color="transparent")
        mid.grid(row=1, column=0, sticky="ew", padx=16, pady=6)
        mid.grid_columnconfigure(0, weight=1)
        mid.grid_columnconfigure(1, weight=1)

        # Kartu Opsi OCR
        ocr_card = make_card(mid)
        ocr_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ocr_card.grid_columnconfigure(1, weight=1)
        section_label(ocr_card, "⚙️  OPSI OCR & PROSES", 0, colspan=2)

        ctk.CTkLabel(ocr_card, text="Bahasa Tesseract:",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=1, column=0, sticky="w", padx=16, pady=5)
        ctk.CTkComboBox(ocr_card, variable=self.v_lang, width=160,
                        values=["ind", "ind+eng", "eng"],
                        font=ctk.CTkFont("Segoe UI", 12)).grid(
            row=1, column=1, sticky="w", padx=(0, 16), pady=5)

        ctk.CTkLabel(ocr_card, text="Format Export:",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=2, column=0, sticky="w", padx=16, pady=5)
        ctk.CTkComboBox(ocr_card, variable=self.v_export, width=160,
                        values=["both", "json", "csv"],
                        font=ctk.CTkFont("Segoe UI", 12)).grid(
            row=2, column=1, sticky="w", padx=(0, 16), pady=5)

        ctk.CTkCheckBox(ocr_card, text="Koreksi kemiringan (Deskew)",
                        variable=self.v_deskew,
                        font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=16, pady=5)
        ctk.CTkCheckBox(ocr_card, text="Mode Dry-Run (preview, file tidak dipindah)",
                        variable=self.v_dryrun,
                        font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=16, pady=(5, 12))

        # Kartu Template Penamaan
        tpl_card = make_card(mid)
        tpl_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tpl_card.grid_columnconfigure(1, weight=1)
        section_label(tpl_card, "🏷️  TEMPLATE PENAMAAN FILE", 0, colspan=2)

        for r, (lbl, var, hint) in enumerate([
            ("Template File:", self.v_file_tpl, "contoh: {jenis}_{nama}"),
            ("Template Folder:", self.v_fol_tpl, "contoh: {JENIS}/{jenis}_{nama}"),
        ], start=1):
            ctk.CTkLabel(tpl_card, text=lbl,
                         font=ctk.CTkFont("Segoe UI", 13)).grid(
                row=r, column=0, sticky="w", padx=16, pady=5)
            ctk.CTkEntry(tpl_card, textvariable=var,
                         placeholder_text=hint,
                         font=ctk.CTkFont("Segoe UI", 12), height=32).grid(
                row=r, column=1, sticky="ew", padx=(0, 16), pady=5)

        ctk.CTkLabel(tpl_card, text="Pemisah Kata:",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=3, column=0, sticky="w", padx=16, pady=5)
        ctk.CTkComboBox(tpl_card, variable=self.v_sep, width=100,
                        values=["_", "-", "."],
                        font=ctk.CTkFont("Segoe UI", 12)).grid(
            row=3, column=1, sticky="w", padx=(0, 16), pady=5)

        # Tooltip variabel template
        tip = ("Variabel tersedia: {jenis} {JENIS} {nama} {NAMA} {nik} {nik6}\n"
               "{tgl} {tgl_compact} {tempat} {jk} {tanggal} {timestamp}\n"
               "KK: {nokk} {nokk6} {kepala} {desa} {kecamatan} {kabupaten} {provinsi}")
        ctk.CTkLabel(tpl_card, text=tip,
                     font=ctk.CTkFont("Segoe UI", 9),
                     text_color=C["gray_text"],
                     justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 10))

        # ── Tombol Aksi ───────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(cfg, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 10))

        self.btn_start = ctk.CTkButton(
            btn_row, text="▶   MULAI PROSES", width=200, height=44,
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color=C["success"], hover_color=C["accent_dark"],
            command=self._on_start)
        self.btn_start.pack(side="left", padx=(0, 10))

        self.btn_stop = ctk.CTkButton(
            btn_row, text="⏹  Berhenti", width=120, height=44,
            font=ctk.CTkFont("Segoe UI", 13),
            fg_color=C["navy"], hover_color=C["navy_mid"],
            state="disabled", command=self._on_stop)
        self.btn_stop.pack(side="left", padx=(0, 10))

        self.btn_report = ctk.CTkButton(
            btn_row, text="🌐 Buka Laporan HTML", width=180, height=44,
            font=ctk.CTkFont("Segoe UI", 13),
            fg_color=C["navy_mid"], hover_color=C["navy"],
            state="disabled", command=self._open_report)
        self.btn_report.pack(side="left")

        self.lbl_status = ctk.CTkLabel(
            btn_row, text="Siap memproses dokumen.",
            font=ctk.CTkFont("Segoe UI", 12), text_color=C["gray_text"])
        self.lbl_status.pack(side="left", padx=16)

    # ── Progress ──────────────────────────────────────────────────────────────
    def _build_progress_area(self):
        pframe = make_card(self._cfg_frame)
        pframe.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 14))
        pframe.grid_columnconfigure(0, weight=1)

        section_label(pframe, "📊  PROGRESS PEMROSESAN", 0)

        self.progress_bar = ctk.CTkProgressBar(pframe, height=18, corner_radius=8,
                                                progress_color=C["accent"])
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 4))
        self.progress_bar.set(0)

        info_row = ctk.CTkFrame(pframe, fg_color="transparent")
        info_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 10))

        self.lbl_prog_count = ctk.CTkLabel(
            info_row, text="0 / 0 file",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=C["accent_dark"])
        self.lbl_prog_count.pack(side="left")

        self.lbl_prog_pct = ctk.CTkLabel(
            info_row, text="0%",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=C["gray_text"])
        self.lbl_prog_pct.pack(side="left", padx=10)

        self.lbl_prog_file = ctk.CTkLabel(
            info_row, text="",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=C["gray_text"])
        self.lbl_prog_file.pack(side="left")

        # Mini stat bar (VALID/ERROR/DRY)
        stat_row = ctk.CTkFrame(pframe, fg_color="transparent")
        stat_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 12))

        self._stat_vals = {}
        for lbl, color in [("✅ VALID", C["success"]),
                            ("❌ ERROR", C["error"]),
                            ("👁️ DRY-RUN", C["warning"])]:
            f = ctk.CTkFrame(stat_row, fg_color="transparent")
            f.pack(side="left", padx=(0, 20))
            ctk.CTkLabel(f, text=lbl,
                         font=ctk.CTkFont("Segoe UI", 11),
                         text_color=color).pack(side="left")
            lv = ctk.CTkLabel(f, text="0",
                               font=ctk.CTkFont("Segoe UI", 11, "bold"),
                               text_color=color)
            lv.pack(side="left", padx=(4, 0))
            key = lbl.split()[-1]  # VALID / ERROR / DRY-RUN
            self._stat_vals[key] = lv
        self._counts = {"VALID": 0, "ERROR": 0, "DRY-RUN": 0}

    # ── Log Area ──────────────────────────────────────────────────────────────
    def _build_log_area(self):
        log_outer = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        log_outer.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))
        log_outer.grid_columnconfigure(0, weight=1)
        log_outer.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(log_outer, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ctk.CTkLabel(hdr, text="📋  Log Output",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=C["text_dark"]).pack(side="left")
        ctk.CTkButton(hdr, text="🗑 Bersihkan", width=90, height=26,
                      font=ctk.CTkFont("Segoe UI", 11),
                      fg_color=C["gray_mid"], text_color=C["text_dark"],
                      hover_color="#D1D5DB",
                      command=self._clear_log).pack(side="right")

        self.log_box = ctk.CTkTextbox(
            log_outer, font=ctk.CTkFont("Consolas", 11),
            corner_radius=8, wrap="word",
            fg_color=("#1E293B", "#111827"),
            text_color="#94A3B8")
        self.log_box.grid(row=1, column=0, sticky="nsew")
        self.log_box.configure(state="disabled")

        # Tag warna log level
        self.log_box._textbox.tag_config("INFO",    foreground="#67E8F9")
        self.log_box._textbox.tag_config("WARNING", foreground="#FDE68A")
        self.log_box._textbox.tag_config("ERROR",   foreground="#FCA5A5")
        self.log_box._textbox.tag_config("DEBUG",   foreground="#9CA3AF")
        self.log_box._textbox.tag_config("OK",      foreground="#86EFAC")

    # ── Event handlers ────────────────────────────────────────────────────────
    def _on_start(self):
        folder = self.v_input.get().strip()
        if not folder:
            messagebox.showerror("Input Kosong",
                "Pilih folder dokumen input terlebih dahulu.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Folder Tidak Ada",
                f"Folder tidak ditemukan:\n{folder}")
            return

        # Reset UI
        self.progress_bar.set(0)
        self.lbl_prog_count.configure(text="0 / ? file")
        self.lbl_prog_pct.configure(text="0%")
        self.lbl_prog_file.configure(text="")
        self._counts = {"VALID": 0, "ERROR": 0, "DRY-RUN": 0}
        for k, lv in self._stat_vals.items():
            lv.configure(text="0")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_report.configure(state="disabled")
        self.lbl_status.configure(text="⏳ Memproses...", text_color=C["warning"])
        self._html_report_path = None
        self._log_lines = 0
        self.append_log(f"{'='*55}")
        self.append_log(f"  Memulai proses: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.append_log(f"  Folder: {folder}")

        params = {
            "input_dir":     folder,
            "lang":          self.v_lang.get(),
            "export_fmt":    self.v_export.get(),
            "file_template": self.v_file_tpl.get() or "{jenis}_{nama}",
            "folder_template": self.v_fol_tpl.get() or "{JENIS}/{jenis}_{nama}",
            "separator":     self.v_sep.get() or "_",
            "dry_run":       self.v_dryrun.get(),
            "use_deskew":    self.v_deskew.get(),
            "ground_truth":  self.v_gt.get().strip() or None,
        }
        self.app.start_batch_process(params)

    def _on_stop(self):
        self.app.request_stop()
        self.lbl_status.configure(text="⏹ Menghentikan...",
                                  text_color=C["error"])
        self.btn_stop.configure(state="disabled")

    def _open_report(self):
        if self._html_report_path and os.path.exists(self._html_report_path):
            webbrowser.open(f"file:///{self._html_report_path}")
        else:
            messagebox.showinfo("Laporan", "File laporan belum tersedia.")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self._log_lines = 0

    # ── Public: dipanggil dari App ────────────────────────────────────────────
    def append_log(self, msg: str):
        """Tambahkan baris log ke textbox (dipanggil dari main thread via after())."""
        self.log_box.configure(state="normal")

        # Tentukan tag warna
        tag = "INFO"
        if "[WARNING]" in msg or "[WARN]" in msg:
            tag = "WARNING"
        elif "[ERROR]" in msg or "[ER]" in msg or "Exception" in msg:
            tag = "ERROR"
        elif "[DEBUG]" in msg:
            tag = "DEBUG"
        elif "[OK]" in msg or "VALID" in msg or "Selesai" in msg:
            tag = "OK"

        self.log_box._textbox.insert("end", msg + "\n", tag)
        self._log_lines += 1

        # Auto-scroll ke bawah
        self.log_box._textbox.see("end")

        # Batasi baris log (performa)
        if self._log_lines > 1500:
            self.log_box._textbox.delete("1.0", "200.0")
            self._log_lines -= 200

        self.log_box.configure(state="disabled")

    def update_progress(self, data: dict):
        """Terima update dari progress_queue dan refresh UI."""
        dtype = data.get("type")

        if dtype == "progress":
            processed = data["processed"]
            total     = data["total"]
            filename  = data.get("filename", "")
            status    = data.get("status", "")

            pct = processed / total if total > 0 else 0
            self.progress_bar.set(pct)
            self.lbl_prog_count.configure(text=f"{processed} / {total} file")
            self.lbl_prog_pct.configure(text=f"{pct*100:.0f}%")
            self.lbl_prog_file.configure(text=f"  {filename[:40]}")

            # Update counter status
            if "VALID" in status and "DRY" not in status:
                self._counts["VALID"] += 1
            elif "DRY" in status:
                self._counts["DRY-RUN"] += 1
            elif "ERROR" in status:
                self._counts["ERROR"] += 1
            for k, v in self._counts.items():
                if k in self._stat_vals:
                    self._stat_vals[k].configure(text=str(v))

        elif dtype == "done":
            self._on_done(data)

        elif dtype == "error":
            self._on_error(data.get("message", "Error tidak diketahui"))

    def _on_done(self, data: dict):
        metrics = data.get("metrics", {})
        html    = data.get("html_path")
        if html:
            self._html_report_path = html
            self.btn_report.configure(state="normal")

        self.progress_bar.set(1.0)
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

        success_rate = metrics.get("success_rate_pct", 0)
        total        = metrics.get("total_files", 0)
        valid        = metrics.get("total_valid", 0)

        self.lbl_status.configure(
            text=f"✅ Selesai — {valid}/{total} berhasil ({success_rate:.1f}%)",
            text_color=C["success"])
        self.append_log("=" * 55)
        self.append_log(f"  ✅ PROSES SELESAI")
        self.append_log(f"  Berhasil: {valid}/{total} ({success_rate:.1f}%)")
        if html:
            self.append_log(f"  Laporan: {html}")

    def _on_error(self, msg: str):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text=f"❌ Error: {msg[:60]}",
                                  text_color=C["error"])
        self.append_log(f"[ERROR] {msg}")
        messagebox.showerror("Error Pemrosesan", msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 2: DEBUG FILE TUNGGAL
# ═══════════════════════════════════════════════════════════════════════════════
class DebugPage(ctk.CTkFrame):
    """Debug satu file: tampilkan semua 12 OCR + hasil ekstraksi field."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.v_file = tk.StringVar()
        self.v_lang = tk.StringVar(value=app.settings.get("default_lang", "ind"))
        self._build()

    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0, height=64, fg_color=C["white"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="🔍  Debug Analisis File Tunggal",
                     font=ctk.CTkFont("Segoe UI", 17, "bold"),
                     text_color=C["navy"]).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        # Config card
        card = make_card(self)
        card.grid(row=1, column=0, sticky="ew", padx=16, pady=14)
        card.grid_columnconfigure(1, weight=1)

        section_label(card, "📄  PILIH FILE DOKUMEN UNTUK DIANALISIS", 0, colspan=3)
        make_browse_row(card, "File Gambar:", self.v_file, 1,
                        file_mode=True, filetypes=SUPPORTED_IMGS,
                        title="Pilih File Dokumen (KTP/KK/SIM)")

        ctk.CTkLabel(card, text="Bahasa OCR:",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=2, column=0, sticky="w", padx=16, pady=8)
        ctk.CTkComboBox(card, variable=self.v_lang, width=160,
                        values=["ind", "ind+eng", "eng"],
                        font=ctk.CTkFont("Segoe UI", 12)).grid(
            row=2, column=1, sticky="w", padx=(0, 16), pady=8)

        # Tombol
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=3, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 14))

        self.btn_run = ctk.CTkButton(
            btn_row, text="🔍  Analisis File", width=160, height=40,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=C["navy"], hover_color=C["navy_mid"],
            command=self._on_run)
        self.btn_run.pack(side="left", padx=(0, 10))

        ctk.CTkButton(btn_row, text="🗑 Bersihkan", width=110, height=40,
                      font=ctk.CTkFont("Segoe UI", 12),
                      fg_color=C["gray_mid"], text_color=C["text_dark"],
                      hover_color="#D1D5DB",
                      command=self._clear).pack(side="left")

        self.lbl_status = ctk.CTkLabel(btn_row, text="",
                                        font=ctk.CTkFont("Segoe UI", 12),
                                        text_color=C["gray_text"])
        self.lbl_status.pack(side="left", padx=12)

        # Output area
        out_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        out_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))
        out_frame.grid_columnconfigure(0, weight=1)
        out_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(out_frame, text="📋  Hasil Analisis OCR",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=C["text_dark"]).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.out_box = ctk.CTkTextbox(
            out_frame, font=ctk.CTkFont("Consolas", 11),
            corner_radius=8, wrap="word",
            fg_color=("#1E293B", "#111827"),
            text_color="#94A3B8")
        self.out_box.grid(row=1, column=0, sticky="nsew")
        self.out_box.configure(state="disabled")

        # Tag warna
        self.out_box._textbox.tag_config("HEAD",  foreground="#FCD34D", font=("Consolas", 11, "bold"))
        self.out_box._textbox.tag_config("FIELD", foreground="#86EFAC")
        self.out_box._textbox.tag_config("STRAT", foreground="#67E8F9")
        self.out_box._textbox.tag_config("WARN",  foreground="#FCA5A5")
        self.out_box._textbox.tag_config("PLAIN", foreground="#94A3B8")

    def _on_run(self):
        fp = self.v_file.get().strip()
        if not fp:
            messagebox.showerror("File Kosong", "Pilih file gambar terlebih dahulu.")
            return
        if not os.path.isfile(fp):
            messagebox.showerror("File Tidak Ada", f"File tidak ditemukan:\n{fp}")
            return
        self.btn_run.configure(state="disabled")
        self.lbl_status.configure(text="⏳ Menganalisis...", text_color=C["warning"])
        self._clear()
        self.append_out("=" * 65, "HEAD")
        self.append_out(f"  DEBUG: {fp}", "HEAD")
        self.append_out("=" * 65, "HEAD")
        self.app.start_debug_process(fp, self.v_lang.get())

    def _clear(self):
        self.out_box.configure(state="normal")
        self.out_box.delete("1.0", "end")
        self.out_box.configure(state="disabled")

    def append_out(self, text: str, tag: str = "PLAIN"):
        self.out_box.configure(state="normal")
        self.out_box._textbox.insert("end", text + "\n", tag)
        self.out_box._textbox.see("end")
        self.out_box.configure(state="disabled")

    def show_debug_result(self, data: dict):
        dtype = data.get("type")
        if dtype == "debug_done":
            output = data.get("output", "")
            self.lbl_status.configure(text="✅ Selesai", text_color=C["success"])
            for line in output.split("\n"):
                if "=" * 5 in line:
                    tag = "HEAD"
                elif any(k in line for k in ["Nama", "NIK", "Kepala", "Lahir",
                                             "Kelamin", "Alamat", "No.", "Nomor",
                                             "Field completeness", "Provinsi",
                                             "Kecamatan", "Kabupaten", "RT/RW",
                                             "Desa"]):
                    tag = "FIELD"
                elif "Strategi" in line or "PSM" in line or "Jenis dokumen" in line:
                    tag = "STRAT"
                elif "ERROR" in line.upper() or "kosong" in line.lower():
                    tag = "WARN"
                else:
                    tag = "PLAIN"
                self.append_out(line, tag)

        elif dtype == "debug_error":
            msg = data.get("message", "Error tidak diketahui")
            self.lbl_status.configure(text=f"❌ Error", text_color=C["error"])
            self.append_out(f"\n[ERROR] {msg}", "WARN")
            messagebox.showerror("Error Debug", msg)

        self.btn_run.configure(state="normal")

    def append_log(self, msg: str):
        pass  # Debug page tidak menampilkan stream log pipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 3: BUAT DEMO
# ═══════════════════════════════════════════════════════════════════════════════
class DemoPage(ctk.CTkFrame):
    """Buat folder demo sintetik untuk pengujian tanpa data nyata."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.v_folder = tk.StringVar(value="Dokumen_Demo")
        self._build()

    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0, height=64, fg_color=C["white"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="🧪  Buat Folder Demo",
                     font=ctk.CTkFont("Segoe UI", 17, "bold"),
                     text_color=C["navy"]).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        # Info card
        info = make_card(self)
        info.grid(row=1, column=0, sticky="ew", padx=16, pady=(16, 8))
        info.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(info,
                     text=(
                         "Mode Demo membuat folder sintetik berisi gambar KTP, KK, SIM, "
                         "dan dokumen tidak dikenal\n"
                         "beserta file ground_truth.csv — berguna untuk verifikasi instalasi "
                         "dan demo skripsi tanpa data asli."
                     ),
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color=C["text_dark"], justify="left",
                     wraplength=700).grid(row=0, column=0, padx=16, pady=12, sticky="w")

        # Isi folder
        detail = make_card(self)
        detail.grid(row=2, column=0, sticky="ew", padx=16, pady=8)
        detail.grid_columnconfigure(0, weight=1)

        section_label(detail, "📁  STRUKTUR FOLDER YANG AKAN DIBUAT", 0)
        tree = (
            "Dokumen_Demo/\n"
            "  ├── batch_A/\n"
            "  │   ├── ktp_dipca.jpg        ← KTP (semua field lengkap)\n"
            "  │   └── kk_sri.jpg           ← KK (Kartu Keluarga)\n"
            "  ├── batch_B/\n"
            "  │   ├── sim_budi.jpg         ← SIM (Surat Izin Mengemudi)\n"
            "  │   └── unknown_doc.jpg      ← Dokumen tidak dikenal → ERROR/\n"
            "  └── ground_truth.csv         ← Data evaluasi F1-Score"
        )
        ctk.CTkLabel(detail, text=tree,
                     font=ctk.CTkFont("Consolas", 12),
                     text_color=C["text_dark"],
                     justify="left").grid(row=1, column=0, padx=16, pady=(4, 12), sticky="w")

        # Form
        form = make_card(self)
        form.grid(row=3, column=0, sticky="ew", padx=16, pady=8)
        form.grid_columnconfigure(1, weight=1)

        section_label(form, "⚙️  PENGATURAN", 0, colspan=2)
        ctk.CTkLabel(form, text="Nama Folder Output:",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=1, column=0, sticky="w", padx=16, pady=8)
        ctk.CTkEntry(form, textvariable=self.v_folder,
                     font=ctk.CTkFont("Segoe UI", 12), height=34).grid(
            row=1, column=1, sticky="ew", padx=(0, 16), pady=8)

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(4, 14))

        self.btn_create = ctk.CTkButton(
            btn_row, text="🧪  Buat Folder Demo", width=180, height=42,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=C["navy"], hover_color=C["navy_mid"],
            command=self._create)
        self.btn_create.pack(side="left", padx=(0, 12))

        self.lbl_status = ctk.CTkLabel(btn_row, text="",
                                        font=ctk.CTkFont("Segoe UI", 12),
                                        text_color=C["gray_text"])
        self.lbl_status.pack(side="left")

        # Output log
        self.log_box = ctk.CTkTextbox(self, font=ctk.CTkFont("Consolas", 11),
                                       corner_radius=8, height=200,
                                       fg_color=("#1E293B", "#111827"),
                                       text_color="#94A3B8")
        self.log_box.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.log_box.configure(state="disabled")

    def _create(self):
        folder = self.v_folder.get().strip() or "Dokumen_Demo"
        self.btn_create.configure(state="disabled")
        self.lbl_status.configure(text="⏳ Membuat...", text_color=C["warning"])

        def worker():
            try:
                from debug_demo import create_demo_structure
                buf = io.StringIO()
                with redirect_stdout(buf):
                    create_demo_structure(folder)
                output = buf.getvalue()
                self.after(0, lambda: self._done(folder, output))
            except Exception as e:
                self.after(0, lambda: self._error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, folder: str, output: str):
        self.btn_create.configure(state="normal")
        self.lbl_status.configure(text=f"✅ Berhasil dibuat: {folder}/",
                                  text_color=C["success"])
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", output)
        self.log_box.configure(state="disabled")
        messagebox.showinfo("Demo Dibuat",
            f"Folder demo berhasil dibuat:\n{os.path.abspath(folder)}\n\n"
            "Gunakan halaman 'Proses Batch' untuk memproses folder ini.")

    def _error(self, msg: str):
        self.btn_create.configure(state="normal")
        self.lbl_status.configure(text="❌ Gagal", text_color=C["error"])
        messagebox.showerror("Error", msg)

    def append_log(self, msg: str):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 4: PENGATURAN
# ═══════════════════════════════════════════════════════════════════════════════
class SettingsPage(ctk.CTkFrame):
    """Konfigurasi global: Tesseract, template default, tema."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)

        s = app.settings
        self.v_tess    = tk.StringVar(value=s.get("tesseract_path", ""))
        self.v_lang    = tk.StringVar(value=s.get("default_lang", "ind"))
        self.v_export  = tk.StringVar(value=s.get("default_export", "both"))
        self.v_ftpl    = tk.StringVar(value=s.get("default_file_template", "{jenis}_{nama}"))
        self.v_dtpl    = tk.StringVar(value=s.get("default_folder_template",
                                                    "{JENIS}/{jenis}_{nama}"))
        self.v_sep     = tk.StringVar(value=s.get("default_separator", "_"))
        self.v_deskew  = tk.BooleanVar(value=s.get("use_deskew", True))
        self.v_theme   = tk.StringVar(value=s.get("appearance_mode", "light"))
        self._build()

    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0, height=64, fg_color=C["white"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="⚙️   Pengaturan Aplikasi",
                     font=ctk.CTkFont("Segoe UI", 17, "bold"),
                     text_color=C["navy"]).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        scroll.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Tesseract ─────────────────────────────────────────────────────────
        tess_card = make_card(scroll)
        tess_card.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        tess_card.grid_columnconfigure(1, weight=1)
        section_label(tess_card, "🔧  TESSERACT OCR", 0, colspan=3)

        ctk.CTkLabel(tess_card,
                     text=("Path ke tesseract.exe (Windows). "
                           "Kosongkan jika Tesseract sudah di PATH sistem."),
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=C["gray_text"]).grid(
            row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 4))

        make_browse_row(tess_card, "Tesseract Path:", self.v_tess, 2,
                        file_mode=True,
                        filetypes=[("Executable", "*.exe"), ("Semua", "*.*")],
                        title="Pilih tesseract.exe")

        self.lbl_tess_ok = ctk.CTkLabel(tess_card, text="",
                                         font=ctk.CTkFont("Segoe UI", 12))
        self.lbl_tess_ok.grid(row=3, column=0, columnspan=3, sticky="w",
                               padx=16, pady=(0, 4))
        ctk.CTkButton(tess_card, text="✔ Cek Tesseract", width=160, height=34,
                      font=ctk.CTkFont("Segoe UI", 12),
                      fg_color=C["navy"], hover_color=C["navy_mid"],
                      command=self._check_tesseract).grid(
            row=4, column=0, sticky="w", padx=16, pady=(0, 12))

        # ── Default OCR ───────────────────────────────────────────────────────
        ocr_card = make_card(scroll)
        ocr_card.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        ocr_card.grid_columnconfigure(1, weight=1)
        section_label(ocr_card, "🔤  DEFAULT OPSI OCR", 0, colspan=2)

        for row_i, (lbl, var, choices) in enumerate([
            ("Bahasa Default:", self.v_lang,   ["ind", "ind+eng", "eng"]),
            ("Export Default:", self.v_export, ["both", "json", "csv"]),
            ("Pemisah Kata:",   self.v_sep,    ["_", "-", "."]),
        ], start=1):
            ctk.CTkLabel(ocr_card, text=lbl,
                         font=ctk.CTkFont("Segoe UI", 13)).grid(
                row=row_i, column=0, sticky="w", padx=16, pady=6)
            ctk.CTkComboBox(ocr_card, variable=var, width=200, values=choices,
                            font=ctk.CTkFont("Segoe UI", 12)).grid(
                row=row_i, column=1, sticky="w", padx=(0, 16), pady=6)

        ctk.CTkCheckBox(ocr_card, text="Aktifkan Deskew secara default",
                        variable=self.v_deskew,
                        font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=16, pady=(6, 12))

        # ── Template default ──────────────────────────────────────────────────
        tpl_card = make_card(scroll)
        tpl_card.grid(row=2, column=0, sticky="ew", padx=16, pady=8)
        tpl_card.grid_columnconfigure(1, weight=1)
        section_label(tpl_card, "🏷️  TEMPLATE PENAMAAN DEFAULT", 0, colspan=2)

        for row_i, (lbl, var, ph) in enumerate([
            ("Template File:",   self.v_ftpl, "{jenis}_{nama}"),
            ("Template Folder:", self.v_dtpl, "{JENIS}/{jenis}_{nama}"),
        ], start=1):
            ctk.CTkLabel(tpl_card, text=lbl,
                         font=ctk.CTkFont("Segoe UI", 13)).grid(
                row=row_i, column=0, sticky="w", padx=16, pady=6)
            ctk.CTkEntry(tpl_card, textvariable=var, placeholder_text=ph,
                         font=ctk.CTkFont("Segoe UI", 12), height=32).grid(
                row=row_i, column=1, sticky="ew", padx=(0, 16), pady=6)

        # ── Tampilan ──────────────────────────────────────────────────────────
        ui_card = make_card(scroll)
        ui_card.grid(row=3, column=0, sticky="ew", padx=16, pady=8)
        ui_card.grid_columnconfigure(1, weight=1)
        section_label(ui_card, "🎨  TAMPILAN APLIKASI", 0, colspan=2)

        ctk.CTkLabel(ui_card, text="Tema Warna:",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
            row=1, column=0, sticky="w", padx=16, pady=8)
        ctk.CTkSegmentedButton(
            ui_card, variable=self.v_theme,
            values=["light", "dark", "system"],
            font=ctk.CTkFont("Segoe UI", 12),
            command=self._change_theme).grid(
            row=1, column=1, sticky="w", padx=(0, 16), pady=8)

        ctk.CTkFrame(ui_card, height=1, fg_color=C["gray_mid"]).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=4)

        # Tombol simpan
        btn_row = ctk.CTkFrame(ui_card, fg_color="transparent")
        btn_row.grid(row=3, column=0, columnspan=2, sticky="w", padx=16, pady=(4, 14))

        ctk.CTkButton(btn_row, text="💾  Simpan Pengaturan", width=180, height=42,
                      font=ctk.CTkFont("Segoe UI", 13, "bold"),
                      fg_color=C["success"], hover_color=C["accent_dark"],
                      command=self._save).pack(side="left", padx=(0, 12))

        ctk.CTkButton(btn_row, text="↺ Reset Default", width=130, height=42,
                      font=ctk.CTkFont("Segoe UI", 12),
                      fg_color=C["gray_mid"], text_color=C["text_dark"],
                      hover_color="#D1D5DB",
                      command=self._reset).pack(side="left")

        self.lbl_saved = ctk.CTkLabel(btn_row, text="",
                                       font=ctk.CTkFont("Segoe UI", 12),
                                       text_color=C["success"])
        self.lbl_saved.pack(side="left", padx=12)

    def _check_tesseract(self):
        import pytesseract
        path = self.v_tess.get().strip()
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
        try:
            ver = pytesseract.get_tesseract_version()
            self.lbl_tess_ok.configure(
                text=f"✅ Tesseract ditemukan: v{ver}",
                text_color=C["success"])
        except Exception as e:
            self.lbl_tess_ok.configure(
                text=f"❌ Tidak ditemukan: {str(e)[:80]}",
                text_color=C["error"])

    def _change_theme(self, val):
        ctk.set_appearance_mode(val)

    def _save(self):
        s = self.app.settings
        s["tesseract_path"]         = self.v_tess.get().strip()
        s["default_lang"]           = self.v_lang.get()
        s["default_export"]         = self.v_export.get()
        s["default_file_template"]  = self.v_ftpl.get().strip()
        s["default_folder_template"]= self.v_dtpl.get().strip()
        s["default_separator"]      = self.v_sep.get()
        s["use_deskew"]             = self.v_deskew.get()
        s["appearance_mode"]        = self.v_theme.get()
        self.app.save_settings()
        self.lbl_saved.configure(text="✅ Tersimpan!")
        self.after(2500, lambda: self.lbl_saved.configure(text=""))

    def _reset(self):
        self.v_tess.set("")
        self.v_lang.set("ind")
        self.v_export.set("both")
        self.v_ftpl.set("{jenis}_{nama}")
        self.v_dtpl.set("{JENIS}/{jenis}_{nama}")
        self.v_sep.set("_")
        self.v_deskew.set(True)
        self.v_theme.set("light")
        ctk.set_appearance_mode("light")

    def append_log(self, msg: str):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 5: LOG VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
class LogPage(ctk.CTkFrame):
    """Tampilkan file log dari folder logs/ pada folder input terakhir."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.v_logdir = tk.StringVar()
        self._build()

    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0, height=64, fg_color=C["white"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="📋  Log Viewer",
                     font=ctk.CTkFont("Segoe UI", 17, "bold"),
                     text_color=C["navy"]).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        # Controls
        ctrl = make_card(self)
        ctrl.grid(row=1, column=0, sticky="ew", padx=16, pady=(14, 8))
        ctrl.grid_columnconfigure(1, weight=1)

        section_label(ctrl, "📁  FOLDER LOG", 0, colspan=3)
        make_browse_row(ctrl, "Folder Log:", self.v_logdir, 1,
                        file_mode=False, title="Pilih Folder logs/")

        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.grid(row=2, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 14))

        ctk.CTkButton(btn_row, text="📂 Muat File Log", width=150, height=38,
                      font=ctk.CTkFont("Segoe UI", 12),
                      fg_color=C["navy"], hover_color=C["navy_mid"],
                      command=self._load_logs).pack(side="left", padx=(0, 10))

        self.log_selector = ctk.CTkComboBox(
            btn_row, width=320, values=["(pilih folder dulu)"],
            font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_select_log)
        self.log_selector.pack(side="left")

        # Textbox
        self.txt = ctk.CTkTextbox(
            self, font=ctk.CTkFont("Consolas", 11),
            corner_radius=8, wrap="word",
            fg_color=("#1E293B", "#111827"),
            text_color="#94A3B8")
        self.txt.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.txt.configure(state="disabled")

    def _load_logs(self):
        folder = self.v_logdir.get().strip()
        if not folder:
            messagebox.showerror("Folder Kosong", "Pilih folder logs/ terlebih dahulu.")
            return
        logs_dir = folder if os.path.basename(folder) == "logs" else os.path.join(folder, "logs")
        if not os.path.isdir(logs_dir):
            messagebox.showinfo("Tidak Ada Log",
                f"Folder log tidak ditemukan:\n{logs_dir}")
            return
        files = sorted([f for f in os.listdir(logs_dir) if f.endswith(".log")],
                       reverse=True)
        if not files:
            messagebox.showinfo("Kosong", "Tidak ada file log.")
            return
        self._log_paths = {f: os.path.join(logs_dir, f) for f in files}
        self.log_selector.configure(values=files)
        self.log_selector.set(files[0])
        self._on_select_log(files[0])

    def _on_select_log(self, choice: str):
        path = getattr(self, "_log_paths", {}).get(choice)
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self.txt.configure(state="normal")
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", content)
            self.txt._textbox.see("end")
            self.txt.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def append_log(self, msg: str):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 6: TENTANG
# ═══════════════════════════════════════════════════════════════════════════════
class AboutPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build()

    def _build(self):
        hdr = ctk.CTkFrame(self, corner_radius=0, height=64, fg_color=C["white"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="ℹ️   Tentang Sistem",
                     font=ctk.CTkFont("Segoe UI", 17, "bold"),
                     text_color=C["navy"]).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=1, column=0, sticky="nsew")
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(0, weight=1)

        card = make_card(center)
        card.grid(row=0, column=0, padx=60, pady=40, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text="🗂️",
                     font=ctk.CTkFont("Segoe UI Emoji", 56)).pack(pady=(32, 8))
        ctk.CTkLabel(card, text=APP_NAME,
                     font=ctk.CTkFont("Segoe UI", 32, "bold"),
                     text_color=C["navy"]).pack()
        ctk.CTkLabel(card, text=APP_SUBTITLE,
                     font=ctk.CTkFont("Segoe UI", 16),
                     text_color=C["gray_text"]).pack(pady=4)
        ctk.CTkLabel(card, text=f"Versi {APP_VERSION}",
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color=C["accent"]).pack(pady=(0, 20))

        ctk.CTkFrame(card, height=1, fg_color=C["gray_mid"]).pack(
            fill="x", padx=40, pady=8)

        info = (
            "Sistem Otomatisasi Penamaan dan Pengarsipan Dokumen Identitas Digital\n\n"
            "Fitur:\n"
            "  • OCR Multi-Strategi: 12 kombinasi preprocessing × PSM\n"
            "  • Klasifikasi otomatis: KTP, KK, SIM\n"
            "  • Ekstraksi field NLP Rule-Based\n"
            "  • Template penamaan file yang dapat dikustomisasi\n"
            "  • Evaluasi Precision/Recall/F1-Score\n"
            "  • Laporan HTML interaktif\n"
            "  • Export metadata JSON + CSV\n\n"
            "Teknologi: Python · pytesseract · OpenCV · Pillow · CustomTkinter\n"
            "OCR Engine: Tesseract-OCR (Language: ind)\n"
        )
        ctk.CTkLabel(card, text=info,
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color=C["text_dark"],
                     justify="left").pack(padx=40, pady=12)

        ctk.CTkFrame(card, height=1, fg_color=C["gray_mid"]).pack(
            fill="x", padx=40, pady=8)
        ctk.CTkLabel(card,
                     text="Dikembangkan untuk keperluan skripsi | Python 3.10+",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=C["gray_text"]).pack(pady=(0, 28))

    def append_log(self, msg: str):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    """Aplikasi utama — mengelola window, navigasi, dan threading."""

    def __init__(self):
        super().__init__()

        # Inisialisasi tema
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} — {APP_SUBTITLE}")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(900, 620)

        # Ikon (opsional, abaikan jika tidak ada)
        try:
            icon = os.path.join(_HERE, "assets", "icon.ico")
            if os.path.isfile(icon):
                self.iconbitmap(icon)
        except Exception:
            pass

        # State
        self.current_page  = None
        self.is_processing = False
        self._stop_flag    = threading.Event()
        self.log_queue     = queue.Queue()
        self.progress_queue= queue.Queue()
        self.config_path   = os.path.join(_HERE, "gui_config.json")
        self.settings      = self._load_settings()

        self._build_layout()
        self._show_page("batch")
        self._poll_queues()

    # ── Konfigurasi persisten ─────────────────────────────────────────────────
    def _load_settings(self) -> dict:
        defaults = {
            "tesseract_path":          "",
            "default_lang":            "ind",
            "default_export":          "both",
            "default_file_template":   "{jenis}_{nama}",
            "default_folder_template": "{JENIS}/{jenis}_{nama}",
            "default_separator":       "_",
            "use_deskew":              True,
            "appearance_mode":         "light",
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        return defaults

    def save_settings(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, self._show_page)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.content = ctk.CTkFrame(self, fg_color=C["gray_light"], corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages: dict[str, ctk.CTkFrame] = {}

    def _show_page(self, page_id: str):
        if self.current_page and self.current_page in self.pages:
            self.pages[self.current_page].grid_remove()

        if page_id not in self.pages:
            cls_map = {
                "batch":    BatchPage,
                "debug":    DebugPage,
                "demo":     DemoPage,
                "settings": SettingsPage,
                "log":      LogPage,
                "about":    AboutPage,
            }
            if page_id in cls_map:
                self.pages[page_id] = cls_map[page_id](self.content, self)

        if page_id in self.pages:
            self.pages[page_id].grid(row=0, column=0, sticky="nsew")
        self.current_page = page_id
        self.sidebar.set_active(page_id)

    # ── Queue polling (main thread) ───────────────────────────────────────────
    def _poll_queues(self):
        # Log queue → halaman batch
        try:
            while True:
                msg = self.log_queue.get_nowait()
                page = self.pages.get("batch")
                if page and hasattr(page, "append_log"):
                    page.append_log(msg)
        except queue.Empty:
            pass

        # Progress queue
        try:
            while True:
                data = self.progress_queue.get_nowait()
                dtype = data.get("type", "")

                if dtype in ("done", "error", "progress"):
                    page = self.pages.get("batch")
                    if page:
                        page.update_progress(data)

                elif dtype in ("debug_done", "debug_error"):
                    page = self.pages.get("debug")
                    if page:
                        page.show_debug_result(data)

        except queue.Empty:
            pass

        self.after(150, self._poll_queues)  # 150ms — cukup responsif, kurangi overhead GUI

    # ── Batch processing ──────────────────────────────────────────────────────
    def start_batch_process(self, params: dict):
        if self.is_processing:
            messagebox.showwarning("Proses Berjalan",
                "Proses masih berjalan. Tunggu hingga selesai.")
            return
        self.is_processing = True
        self._stop_flag.clear()
        t = threading.Thread(target=self._batch_worker, args=(params,), daemon=True)
        t.start()

    def request_stop(self):
        self._stop_flag.set()

    def _batch_worker(self, params: dict):
        try:
            import pytesseract
            from pipeline import process_folder

            tess = params.get("tesseract_path") or self.settings.get("tesseract_path")
            if tess:
                pytesseract.pytesseract.tesseract_cmd = tess

            # Setup logging ke GUI queue
            from logger import setup_logger
            log_inst = setup_logger(params["input_dir"])
            q_handler = _QueueHandler(self.log_queue)
            q_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
            log_inst.addHandler(q_handler)

            def progress_cb(processed, total, filename, status):
                if self._stop_flag.is_set():
                    raise InterruptedError("Proses dihentikan oleh pengguna.")
                self.progress_queue.put({
                    "type":      "progress",
                    "processed": processed,
                    "total":     total,
                    "filename":  filename,
                    "status":    status,
                })

            metrics = process_folder(
                root_input        = params["input_dir"],
                lang              = params["lang"],
                export_fmt        = params["export_fmt"],
                ground_truth_path = params.get("ground_truth"),
                file_template     = params["file_template"],
                folder_template   = params["folder_template"],
                sep               = params["separator"],
                dry_run           = params["dry_run"],
                use_deskew        = params["use_deskew"],
                progress_callback = progress_cb,
                stop_event        = self._stop_flag,  # ← teruskan stop event
            )

            # Cari file laporan HTML terbaru
            meta_dir  = os.path.join(params["input_dir"], "metadata")
            html_path = None
            if os.path.isdir(meta_dir):
                htmls = sorted(
                    [f for f in os.listdir(meta_dir) if f.startswith("laporan_")
                     and f.endswith(".html")],
                    reverse=True
                )
                if htmls:
                    html_path = os.path.join(meta_dir, htmls[0])

            self.progress_queue.put({
                "type":      "done",
                "metrics":   metrics,
                "html_path": html_path,
            })

        except InterruptedError as e:
            self.progress_queue.put({"type": "error", "message": str(e)})
        except Exception as e:
            self.progress_queue.put({"type": "error", "message": str(e)})
        finally:
            self.is_processing = False

    # ── Debug file tunggal ────────────────────────────────────────────────────
    def start_debug_process(self, file_path: str, lang: str):
        if self.is_processing:
            return
        self.is_processing = True
        t = threading.Thread(target=self._debug_worker,
                             args=(file_path, lang), daemon=True)
        t.start()

    def _debug_worker(self, file_path: str, lang: str):
        try:
            import pytesseract
            tess = self.settings.get("tesseract_path")
            if tess:
                pytesseract.pytesseract.tesseract_cmd = tess

            from debug_demo import debug_single_file
            buf = io.StringIO()
            with redirect_stdout(buf):
                debug_single_file(file_path, lang=lang)
            self.progress_queue.put({"type": "debug_done", "output": buf.getvalue()})
        except Exception as e:
            self.progress_queue.put({"type": "debug_error", "message": str(e)})
        finally:
            self.is_processing = False


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Periksa CustomTkinter tersedia
    try:
        import customtkinter  # noqa
    except ImportError:
        print("\n[ERROR] customtkinter tidak terinstal.")
        print("  Instal dengan: pip install customtkinter")
        sys.exit(1)

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()