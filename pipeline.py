"""
pipeline.py — Pipeline Utama Batch Processing
==============================================
Mengintegrasikan semua modul dalam satu alur kerja batch lengkap (Modul 10).

Perubahan v5.7-GUI:
  - Tambah parameter progress_callback opsional di process_folder()
  - Callback dipanggil setelah setiap file: callback(processed, total, filename, status)
  - Sepenuhnya backward-compatible dengan CLI lama

Optimasi v5.7-Perf:
  - File-file diproses secara paralel via ThreadPoolExecutor (FILE_WORKERS worker)
  - Stop flag diperiksa sebelum setiap file dimulai
  - Thread-safe progress counter dengan threading.Lock
"""
import os, re, shutil, time, logging, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Optional

try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from config import (
    SUPPORTED_EXT, SKIP_DIRS,
    DEFAULT_FILE_TEMPLATE, DEFAULT_FOLDER_TEMPLATE, DEFAULT_SEPARATOR,
)
from logger import setup_logger
from preprocessing import extract_text, clean_text
from extractor import classify_document, extract_fields, calculate_field_confidence
from file_manager import build_output_path, move_file
from evaluator import (
    export_metadata,
    evaluate_results, print_evaluation,
    evaluate_naming_convention, export_naming_report,
    evaluate_time_efficiency, print_time_report,
    load_ground_truth, calculate_f1_score, print_f1_report, export_f1_report,
    export_html_report, _save_metrics,
)

log = logging.getLogger("otomatisasi_dokumen")

# Jumlah worker untuk proses file paralel.
# Gunakan 2-4 untuk laptop; 4-6 untuk workstation. Jangan terlalu tinggi karena
# Tesseract sendiri sudah multi-thread via OCR_WORKERS di preprocessing.py.
FILE_WORKERS = 4


def process_folder(
    root_input         : str,
    lang               : str  = "ind",
    export_fmt         : str  = "both",
    ground_truth_path  : str  = None,
    file_template      : str  = DEFAULT_FILE_TEMPLATE,
    folder_template    : str  = DEFAULT_FOLDER_TEMPLATE,
    sep                : str  = DEFAULT_SEPARATOR,
    dry_run            : bool = False,
    use_deskew         : bool = True,
    progress_callback  : Optional[Callable] = None,
    stop_event         : Optional[threading.Event] = None,
) -> dict:
    """
    Pipeline utama — memproses semua gambar dalam root_input secara batch paralel.

    Args:
        progress_callback: Opsional, untuk GUI. Signature:
                           callback(processed: int, total: int, filename: str, status: str)
        stop_event:        Opsional, threading.Event untuk menghentikan proses lebih awal.
    """
    global log
    log          = setup_logger(root_input)
    batch_start  = time.time()
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 60)
    log.info("  SISTEM OTOMATISASI PENAMAAN DAN PENGARSIPAN DOKUMEN IDENTITAS DIGITAL")
    log.info(f"  Folder input  : {os.path.abspath(root_input)}")
    log.info(f"  Bahasa OCR    : {lang}")
    log.info(f"  Export format : {export_fmt.upper()}")
    log.info(f"  Template file  : {file_template}")
    log.info(f"  Template folder: {folder_template}")
    log.info(f"  Separator      : '{sep}'")
    log.info(f"  Dry-run        : {'YA (preview, file tidak dipindah)' if dry_run else 'Tidak'}")
    log.info(f"  Deskewing      : {'Aktif' if use_deskew else 'Nonaktif'}")
    if ground_truth_path:
        log.info(f"  Ground truth   : {ground_truth_path}")
    log.info(f"  Mulai          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    if dry_run:
        log.info("")
        log.info("  [DRY-RUN] File TIDAK akan dipindahkan.")
        log.info("  Hapus flag --dry-run untuk proses sebenarnya.")
        log.info("")

    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_input):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT:
                all_files.append((dirpath, fn))

    total_files = len(all_files)
    log.info(f"  Total file ditemukan: {total_files}")
    records = []

    use_gui_callback = progress_callback is not None
    if not use_gui_callback and TQDM_AVAILABLE and total_files > 0:
        pbar = _tqdm(
            total=total_files, desc="  Memproses", unit="file",
            bar_format="{l_bar}{bar:28}{r_bar}", ncols=75,
        )
    else:
        pbar = None

    processed = 0
    _lock = threading.Lock()   # Lindungi counter processed dari race condition

    def _process_and_report(dirpath, filename):
        """Worker: proses satu file lalu laporkan progress (thread-safe)."""
        nonlocal processed

        # Cek stop flag sebelum mulai
        if stop_event and stop_event.is_set():
            return None

        rel_dir  = os.path.relpath(dirpath, root_input)
        src_path = os.path.join(dirpath, filename)
        log.debug(f"\n  > {filename}")

        record = _process_single(
            src_path=src_path, root_input=root_input, lang=lang,
            rel_dir=rel_dir, file_template=file_template,
            folder_template=folder_template, sep=sep,
            dry_run=dry_run, use_deskew=use_deskew,
        )

        with _lock:
            processed += 1
            current   = processed

        st   = record["status"]
        icon = "[OK]" if "VALID" in st else "[--]" if "DRY" in st else "[ER]"

        if use_gui_callback:
            # Cek stop flag lagi setelah proses selesai
            if stop_event and stop_event.is_set():
                return record
            progress_callback(current, total_files, filename, st)
        elif pbar:
            with _lock:
                pbar.set_postfix_str(f"{icon} {filename[:22]}", refresh=True)
                pbar.update(1)
        else:
            with _lock:
                pct = int(current / total_files * 28)
                bar = "[" + "#" * pct + "." * (28 - pct) + "]"
                print(f"\r  {bar} {current}/{total_files}  {filename[:20]}",
                      end="", flush=True)
        return record

    # Proses semua file secara paralel dengan FILE_WORKERS worker
    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
        futures = [
            executor.submit(_process_and_report, dirpath, filename)
            for dirpath, filename in all_files
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                records.append(result)

    if pbar:
        pbar.close()
    elif total_files > 0 and not use_gui_callback:
        print()
    log.info("")

    export_metadata(records, root_input, fmt=export_fmt, timestamp=timestamp)
    metrics = evaluate_results(records, batch_start_time=batch_start)
    print_evaluation(metrics, log)
    naming_result = evaluate_naming_convention(records)
    export_naming_report(naming_result, root_input, timestamp)
    log.info(
        f"  Naming compliance: {naming_result['compliance_pct']}% "
        f"({naming_result['valid']} sesuai, {naming_result['invalid']} tidak sesuai)"
    )
    time_result = evaluate_time_efficiency(records, batch_start)
    print_time_report(time_result, log)

    if ground_truth_path:
        try:
            gt        = load_ground_truth(ground_truth_path)
            f1_result = calculate_f1_score(records, gt)
            print_f1_report(f1_result, log)
            export_f1_report(f1_result, root_input, timestamp)
            metrics["f1_evaluation"] = f1_result
        except (FileNotFoundError, ValueError) as e:
            log.error(str(e))

    metrics["time_efficiency"]   = time_result
    metrics["naming_compliance"] = naming_result
    _save_metrics(metrics, root_input, timestamp)

    html_path = export_html_report(records, metrics, root_input, timestamp)
    log.info(f"  Buka laporan: file://{os.path.abspath(html_path)}")
    mode_tag = " [DRY-RUN]" if dry_run else ""
    log.info(
        f"  Selesai{mode_tag}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Durasi: {time_result['total_batch_detik']} detik"
    )
    return metrics


def _init_record(filename: str, rel_dir: str, dry_run: bool) -> dict:
    return {
        "nama_file_asli": filename, "nama_file_baru": "", "subfolder": rel_dir,
        "jenis_dokumen": "unknown", "status": "ERROR", "path_tujuan": "",
        "nama": "", "nik": "", "tempat_lahir": "", "tanggal_lahir": "",
        "jenis_kelamin": "", "alamat": "", "nomor_kk": "", "nama_kepala": "",
        "desa_kelurahan": "", "kecamatan": "", "kabupaten_kota": "", "provinsi": "",
        "rtrw": "", "no_sim": "", "field_completeness": 0.0,
        "fields_filled": 0, "fields_total": 0, "waktu_proses": "",
        "confidence_scores": {}, "dry_run": dry_run,
    }


def _update_record_fields(record: dict, doc_type: str, fields: dict) -> None:
    record.update({
        "nama": fields["nama"], "nik": fields["nik"],
        "tempat_lahir": fields["tempat_lahir"], "tanggal_lahir": fields["tanggal_lahir"],
        "jenis_kelamin": fields["jenis_kelamin"], "alamat": fields["alamat"],
        "field_completeness": fields["_field_completeness"],
        "fields_filled": fields["_fields_filled"], "fields_total": fields["_fields_total"],
    })
    if doc_type == "kk":
        record.update({
            "nomor_kk": fields.get("nomor_kk", ""), "nama_kepala": fields.get("nama_kepala", ""),
            "desa_kelurahan": fields.get("desa_kelurahan", ""), "kecamatan": fields.get("kecamatan", ""),
            "kabupaten_kota": fields.get("kabupaten_kota", ""), "provinsi": fields.get("provinsi", ""),
            "rtrw": fields.get("rtrw", ""),
        })
    if doc_type == "sim":
        record.update({"no_sim": fields.get("no_sim", "")})


def _log_fields(doc_type: str, fields: dict) -> None:
    if doc_type == "kk":
        log.info(f"  Nomor KK        : {fields.get('nomor_kk') or '-'}")
        log.info(f"  Kepala Keluarga : {fields.get('nama_kepala') or '-'}")
        log.info(f"  Desa/Kelurahan  : {fields.get('desa_kelurahan') or '-'}")
        log.info(f"  Kecamatan       : {fields.get('kecamatan') or '-'}")
        log.info(f"  Kabupaten/Kota  : {fields.get('kabupaten_kota') or '-'}")
        log.info(f"  Provinsi        : {fields.get('provinsi') or '-'}")
        log.info(f"  RT/RW           : {fields.get('rtrw') or '-'}")
        log.info(f"  Alamat          : {fields.get('alamat') or '-'}")
    elif doc_type == "sim":
        log.info(f"  No. SIM         : {fields.get('no_sim') or '-'}")
        log.info(f"  Nama            : {fields['nama'] or '-'}")
        log.info(f"  Tempat Lahir    : {fields['tempat_lahir'] or '-'}")
        log.info(f"  Tanggal Lahir   : {fields['tanggal_lahir'] or '-'}")
        log.info(f"  Jenis Kelamin   : {fields['jenis_kelamin'] or '-'}")
    else:
        log.info(f"  Nama            : {fields['nama'] or '-'}")
        log.info(f"  NIK             : {fields['nik'] or '-'}")
        log.info(f"  Tempat Lahir    : {fields['tempat_lahir'] or '-'}")
        log.info(f"  Tanggal Lahir   : {fields['tanggal_lahir'] or '-'}")
        log.info(f"  Jenis Kelamin   : {fields['jenis_kelamin'] or '-'}")
    log.info(
        f"  Field completeness: "
        f"{fields['_fields_filled']}/{fields['_fields_total']} "
        f"({fields['_field_completeness'] * 100:.0f}%)"
    )


def _move_or_preview(record, src_path, dest_path, root_input, success, dry_run):
    if dry_run:
        log.info(f"  [DRY-RUN] Preview: {os.path.relpath(dest_path, root_input)}")
        record.update({
            "nama_file_baru": os.path.basename(dest_path),
            "status": "DRY-RUN-VALID" if success else "DRY-RUN-ERROR",
            "path_tujuan": os.path.relpath(dest_path, root_input),
        })
    else:
        move_file(src_path, dest_path)
        record.update({
            "nama_file_baru": os.path.basename(dest_path),
            "status": "VALID" if success else "ERROR",
            "path_tujuan": os.path.relpath(dest_path, root_input),
        })


def _process_single(src_path, root_input, lang, rel_dir,
                    file_template=DEFAULT_FILE_TEMPLATE,
                    folder_template=DEFAULT_FOLDER_TEMPLATE,
                    sep=DEFAULT_SEPARATOR, dry_run=False, use_deskew=True):
    filename   = os.path.basename(src_path)
    start_time = datetime.now()
    record     = _init_record(filename, rel_dir, dry_run)
    try:
        best_text, all_texts = extract_text(src_path, lang=lang, use_deskew=use_deskew)
        if not best_text.strip():
            log.warning("  OCR kosong — pastikan gambar tidak buram/gelap.")
            _move_to_error(src_path, root_input, rel_dir)
            record["waktu_proses"] = _elapsed(start_time)
            return record
        clean     = clean_text(best_text)
        all_clean = [clean_text(t) for t in all_texts]
        doc_type                = classify_document(clean)
        record["jenis_dokumen"] = doc_type
        log.info(f"  Jenis dokumen   : {doc_type.upper()}")
        fields = extract_fields(clean, doc_type, all_clean)
        _update_record_fields(record, doc_type, fields)
        _log_fields(doc_type, fields)
        conf = calculate_field_confidence(all_clean, doc_type)
        record["confidence_scores"] = conf
        if conf:
            avg_c = round(sum(conf.values()) / len(conf), 1)
            log.info(f"  Avg confidence    : {avg_c}%")
        name_slug = fields["nama"]
        success   = doc_type != "unknown" and bool(name_slug)
        dest_path = build_output_path(
            src_path=src_path, root_input=root_input, doc_type=doc_type,
            name=name_slug, success=success, fields=fields,
            file_template=file_template, folder_template=folder_template, sep=sep,
        )
        _move_or_preview(record, src_path, dest_path, root_input, success, dry_run)
    except Exception as exc:
        log.error(f"  Exception: {exc}", exc_info=True)
        try:
            _move_to_error(src_path, root_input, rel_dir)
        except Exception as move_err:
            log.debug(f"  Gagal pindah ke ERROR/: {move_err}")
    record["waktu_proses"] = _elapsed(start_time)
    return record


def _move_to_error(src_path: str, root_input: str, rel_dir: str) -> None:
    filename  = os.path.basename(src_path)
    error_dir = os.path.join(root_input, "ERROR", rel_dir)
    os.makedirs(error_dir, exist_ok=True)
    dest    = os.path.join(error_dir, filename)
    counter = 1
    while os.path.exists(dest):
        base, ext = os.path.splitext(filename)
        dest      = os.path.join(error_dir, f"{base}_{counter}{ext}")
        counter  += 1
    shutil.copy2(src_path, dest)
    os.remove(src_path)


def _elapsed(start: datetime) -> str:
    delta = datetime.now() - start
    ms    = int(delta.total_seconds() * 1000)
    return f"{ms} ms"


def _save_metrics(metrics: dict, root_input: str, timestamp: str = None) -> None:
    meta_dir = os.path.join(root_input, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(meta_dir, f"evaluasi_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info(f"Metrik evaluasi: {path}")
