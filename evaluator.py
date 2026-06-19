"""
evaluator.py — Evaluasi Kinerja, Ekspor Metadata, dan Laporan HTML
===================================================================
Menangani:
  - Laporan HTML interaktif (Modul 1F)
  - Ekspor metadata JSON + CSV (Modul 8)
  - Evaluasi statistik sistem: success rate, distribusi, field completeness (Modul 9A)
  - Evaluasi konsistensi penamaan file / naming convention (Modul 9B)
  - Evaluasi Precision / Recall / F1-Score per field vs ground truth (Modul 9C)
  - Evaluasi efisiensi waktu vs estimasi proses manual (Modul 9D)
  - Simpan semua metrik ke JSON (_save_metrics)
"""
import os, re, csv, json, sys, time, logging
from datetime import datetime

from config import VERSION, NAMING_PATTERN, MANUAL_TIME_SECONDS

log = logging.getLogger("otomatisasi_dokumen")

def _html_rows(records: list) -> str:
    """Bangun baris <tr> tabel dokumen untuk laporan HTML."""
    rows = []
    for r in records:
        st    = r.get("status", "ERROR")
        bg    = "#e8f5e9" if "VALID" in st else "#fff8e1" if "DRY" in st else "#ffebee"
        bdg   = ('<span class="badge ok">VALID</span>' if st == "VALID"
                 else '<span class="badge dry">PREVIEW</span>' if "DRY" in st
                 else '<span class="badge err">ERROR</span>')
        conf  = r.get("confidence_scores", {})
        cavg  = round(sum(conf.values()) / len(conf)) if conf else None
        cbar  = (f'<div class="cbar"><div class="cfill" style="width:{cavg}%"></div>'
                 f'</div><span class="cpct">{cavg}%</span>' if cavg is not None else "—")
        fc    = round(r.get("field_completeness", 0) * 100)
        jenis = r.get("jenis_dokumen", "")

        if jenis == "kk":
            col_nama   = r.get("nama_kepala", "") or r.get("nama", "")
            col_nik    = r.get("nomor_kk", "") or r.get("nik", "")
            kk_details = []
            if r.get("desa_kelurahan"): kk_details.append(f"Kel: {r['desa_kelurahan']}")
            if r.get("kecamatan"):      kk_details.append(f"Kec: {r['kecamatan']}")
            if r.get("kabupaten_kota"): kk_details.append(f"Kab: {r['kabupaten_kota']}")
            if r.get("provinsi"):       kk_details.append(f"Prov: {r['provinsi']}")
            if r.get("rtrw"):           kk_details.append(f"RT/RW: {r['rtrw']}")
            extra = (f'<br><small style="color:#666">{" | ".join(kk_details)}</small>'
                     if kk_details else "")
        elif jenis == "sim":
            col_nama = r.get("nama", "")
            no_sim   = r.get("no_sim", "")
            col_nik  = no_sim if no_sim else r.get("nik", "")
            extra    = (f'<br><small style="color:#e65100">No. SIM: {no_sim}</small>'
                        if no_sim else "")
        else:
            col_nama = r.get("nama", "")
            col_nik  = r.get("nik", "")
            extra    = ""

        rows.append(
            f'<tr style="background:{bg}">'
            f'<td>{r.get("nama_file_asli","")}</td>'
            f'<td>{r.get("nama_file_baru","")}</td>'
            f'<td class="center"><b>{jenis.upper()}</b></td>'
            f'<td class="center">{bdg}</td>'
            f'<td>{col_nama}{extra}</td>'
            f'<td>{col_nik}</td>'
            f'<td class="center">{fc}%</td>'
            f'<td class="center">{cbar}</td>'
            f'<td class="center">{r.get("waktu_proses","")}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _html_dist(dist: dict, total: int) -> str:
    """Bangun blok distribusi jenis dokumen untuk laporan HTML."""
    colors = {"ktp": "#1976d2", "kk": "#388e3c", "sim": "#f57c00", "unknown": "#757575"}
    parts  = []
    for jenis in ["ktp", "kk", "sim", "unknown"]:
        cnt = dist.get(jenis, 0)
        if not cnt:
            continue
        pct = round(cnt / total * 100) if total else 0
        c   = colors.get(jenis, "#607d8b")
        parts.append(
            f'<div class="dist-row">'
            f'<span class="dist-lbl" style="color:{c}">{jenis.upper()}</span>'
            f'<div class="dist-track"><div class="dist-fill" '
            f'style="background:{c};width:{pct * 2}px"></div></div>'
            f'<span class="dist-val">{cnt} dok ({pct}%)</span>'
            f'</div>'
        )
    return "\n".join(parts)


def _html_f1_section(f1_r: dict) -> str:
    """Bangun seksi tabel F1-Score untuk laporan HTML (kosong jika tidak ada GT)."""
    if not f1_r or not f1_r.get("per_field"):
        return ""
    labels = {
        "nama": "Nama", "nik": "NIK", "tempat_lahir": "Tempat Lahir",
        "tanggal_lahir": "Tanggal Lahir", "jenis_kelamin": "Jenis Kelamin",
        "jenis_dokumen": "Jenis Dokumen", "alamat": "Alamat",
        "nomor_kk": "Nomor KK", "nama_kepala": "Nama Kepala",
        "desa_kelurahan": "Desa/Kelurahan", "kecamatan": "Kecamatan",
        "kabupaten_kota": "Kab/Kota", "provinsi": "Provinsi",
        "rtrw": "RT/RW", "no_sim": "No SIM",
    }
    rows = []
    for fld, lbl in labels.items():
        m = f1_r["per_field"].get(fld, {})
        if not m or (m.get("tp", 0) + m.get("fp", 0) + m.get("fn", 0)) == 0:
            continue
        f1v = m.get("f1_score", 0)
        rows.append(
            f'<tr><td>{lbl}</td>'
            f'<td class="center">{m.get("precision", 0):.2f}</td>'
            f'<td class="center">{m.get("recall", 0):.2f}</td>'
            f'<td class="center"><div class="cbar"><div class="cfill" '
            f'style="width:{int(f1v * 100)}%;background:#9c27b0"></div></div>'
            f'<span class="cpct">{f1v:.2f}</span></td>'
            f'<td class="center">{m.get("tp", 0)}</td>'
            f'<td class="center">{m.get("fp", 0)}</td>'
            f'<td class="center">{m.get("fn", 0)}</td></tr>'
        )
    mf1 = f1_r.get("macro_f1_score", 0)
    unmatched = f1_r.get("dokumen_tidak_tercocok", 0)
    unmatched_note = (
        f'<p class="sub" style="color:#c62828">'
        f'⚠ {unmatched} dokumen tidak ter-match (dihitung sebagai FN pada seluruh field)'
        f'</p>'
    ) if unmatched > 0 else ""
    rows.append(
        f'<tr style="font-weight:bold;background:#f3e5f5"><td>MACRO AVG</td>'
        f'<td class="center">{f1_r.get("macro_precision", 0):.2f}</td>'
        f'<td class="center">{f1_r.get("macro_recall", 0):.2f}</td>'
        f'<td class="center"><div class="cbar"><div class="cfill" '
        f'style="width:{int(mf1 * 100)}%;background:#9c27b0"></div></div>'
        f'<span class="cpct">{mf1:.2f}</span></td>'
        f'<td class="center">—</td><td class="center">—</td><td class="center">—</td></tr>'
    )
    return (
        '<div class="section">'
        '<h2 style="color:#6a1b9a">Evaluasi Precision / Recall / F1-Score</h2>'
        f'<p class="sub">Ground truth: {f1_r.get("total_ground_truth", 0)} dokumen dievaluasi '
        f'({f1_r.get("dokumen_dievaluasi", 0)} ter-match).</p>'
        f'{unmatched_note}'
        '<table><thead><tr><th>Field</th><th>Precision</th><th>Recall</th>'
        '<th>F1-Score</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _html_waktu_section(time_r: dict) -> str:
    """Bangun seksi efisiensi waktu untuk laporan HTML."""
    if not time_r:
        return ""
    ratio = time_r.get("rasio_kecepatan", 0)
    clr   = "#2e7d32" if ratio >= 5 else "#f57f17"
    return (
        '<div class="section">'
        '<h2 style="color:#e65100">Efisiensi Waktu Proses</h2>'
        '<table><tbody>'
        f'<tr><td>Rata-rata per dokumen</td><td><b>{time_r.get("avg_ms", 0)} ms</b></td></tr>'
        f'<tr><td>Tercepat / Terlama</td>'
        f'<td>{time_r.get("min_ms", 0)} ms / {time_r.get("max_ms", 0)} ms</td></tr>'
        f'<tr><td>Total waktu batch</td><td><b>{time_r.get("total_batch_detik", 0)} dtk</b></td></tr>'
        f'<tr><td>Estimasi waktu manual</td><td>{time_r.get("estimasi_manual_detik", 0)} dtk</td></tr>'
        f'<tr><td>Sistem lebih cepat</td>'
        f'<td><b style="color:{clr};font-size:18px">{ratio}x</b></td></tr>'
        '</tbody></table></div>'
    )


def _html_naming_section(naming: dict) -> str:
    """Bangun seksi konsistensi penamaan untuk laporan HTML."""
    if not naming:
        return ""
    nc     = naming.get("compliance_pct", 0)
    nc_clr = "#2e7d32" if nc >= 90 else "#f57f17" if nc >= 70 else "#c62828"
    return (
        '<div class="section">'
        '<h2 style="color:#0277bd">Konsistensi Penamaan File</h2>'
        f'<p>Compliance: <b style="color:{nc_clr};font-size:22px">{nc}%</b>'
        f'&nbsp;&nbsp;({naming.get("valid", 0)} sesuai, '
        f'{naming.get("invalid", 0)} tidak sesuai)</p></div>'
    )


def _html_css() -> str:
    """Kembalikan string CSS inline untuk laporan HTML."""
    return """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#333;padding:20px}
.wrap{max-width:1100px;margin:0 auto}
.header{background:linear-gradient(135deg,#1565c0,#0d47a1);color:#fff;
        padding:24px 30px;border-radius:12px;margin-bottom:20px}
.header h1{font-size:20px;font-weight:700;margin-bottom:4px}
.header p{font-size:12px;opacity:.85}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
       gap:14px;margin-bottom:20px}
.card{background:#fff;border-radius:10px;padding:18px;text-align:center;
      box-shadow:0 2px 6px rgba(0,0,0,.07)}
.card .v{font-size:34px;font-weight:700;line-height:1.1}
.card .l{font-size:11px;color:#888;margin-top:3px}
.section{background:#fff;border-radius:10px;padding:22px;margin-bottom:16px;
         box-shadow:0 2px 6px rgba(0,0,0,.06)}
h2{font-size:15px;margin-bottom:12px}
.sub{font-size:12px;color:#666;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px}
th{background:#1565c0;color:#fff;padding:8px 10px;text-align:left}
td{padding:7px 10px;border-bottom:1px solid #f0f0f0}
tr:hover td{background:rgba(21,101,192,.03)}
.center{text-align:center}
.badge{padding:2px 8px;border-radius:10px;font-size:11px;color:#fff;font-weight:600}
.badge.ok{background:#4caf50}.badge.err{background:#f44336}
.badge.dry{background:#ff9800}
.cbar{display:inline-block;background:#e0e0e0;border-radius:4px;
      height:8px;width:80px;vertical-align:middle;margin-right:4px}
.cfill{height:8px;border-radius:4px;background:#2196f3}
.cpct{font-size:11px;color:#555}
.dist-row{display:flex;align-items:center;margin:5px 0;gap:10px}
.dist-lbl{width:44px;font-weight:700;font-size:13px}
.dist-track{background:#e0e0e0;border-radius:6px;height:16px;width:200px}
.dist-fill{height:16px;border-radius:6px}
.dist-val{font-size:12px;color:#555}
#searchBox{padding:7px 12px;border:1px solid #ddd;border-radius:6px;
           width:280px;font-size:12px;margin-bottom:10px}
footer{text-align:center;color:#bbb;font-size:11px;margin-top:20px}"""


def export_html_report(
    records     : list,
    metrics     : dict,
    output_root : str,
    timestamp   : str,
) -> str:
    """
    Buat laporan HTML interaktif dari hasil pemrosesan batch.

    Laporan di-embed sepenuhnya dalam satu file HTML (inline CSS + JS)
    sehingga portable. Perakitan HTML didelegasikan ke sub-fungsi:
      _html_rows()           — baris tabel dokumen
      _html_dist()           — distribusi jenis dokumen
      _html_f1_section()     — tabel F1-Score (jika ada ground truth)
      _html_waktu_section()  — efisiensi waktu
      _html_naming_section() — konsistensi penamaan
      _html_css()            — string CSS inline

    Args:
        records     (list): Daftar record hasil tiap dokumen.
        metrics     (dict): Output evaluate_results() + modul evaluasi lain.
        output_root (str) : Folder root untuk subfolder metadata/.
        timestamp   (str) : Timestamp untuk nama file.

    Returns:
        str: Path absolut file HTML yang dibuat.
    """
    meta_dir  = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    html_path = os.path.join(meta_dir, f"laporan_{timestamp}.html")

    total  = metrics.get("total_dokumen", 0)
    valid  = metrics.get("dokumen_valid", 0)
    error  = metrics.get("dokumen_error", 0)
    rate   = metrics.get("success_rate_pct", 0)
    compl  = metrics.get("avg_field_completeness", 0) * 100
    dist   = metrics.get("distribusi_jenis", {})
    tgl    = datetime.now().strftime("%d %B %Y %H:%M")

    rate_clr      = "#2e7d32" if rate >= 90 else "#f57f17" if rate >= 70 else "#c62828"
    rows_html     = _html_rows(records)
    dist_html     = _html_dist(dist, total)
    f1_section    = _html_f1_section(metrics.get("f1_evaluation", {}))
    waktu_section = _html_waktu_section(metrics.get("time_efficiency", {}))
    naming_section = _html_naming_section(metrics.get("naming_compliance", {}))
    css           = _html_css()

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laporan Otomatisasi Penamaan Dokumen Identitas Digital {tgl}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">

<div class="header">
  <h1>Laporan Sistem Otomatisasi Penamaan dan Pengarsipan Dokumen Identitas Digital</h1>
  <p>Dibuat: {tgl} &nbsp;|&nbsp; Folder: <code>{os.path.abspath(output_root)}</code></p>
</div>

<div class="cards">
  <div class="card"><div class="v" style="color:#1565c0">{total}</div>
    <div class="l">Total Dokumen</div></div>
  <div class="card"><div class="v" style="color:#2e7d32">{valid}</div>
    <div class="l">Berhasil (VALID)</div></div>
  <div class="card"><div class="v" style="color:#c62828">{error}</div>
    <div class="l">Gagal (ERROR)</div></div>
  <div class="card"><div class="v" style="color:{rate_clr}">{rate}%</div>
    <div class="l">Success Rate</div></div>
  <div class="card"><div class="v" style="color:#6a1b9a">{compl:.0f}%</div>
    <div class="l">Avg Field Compl.</div></div>
</div>

<div class="section">
  <h2 style="color:#1565c0">Distribusi Jenis Dokumen</h2>
  {dist_html}
</div>

<div class="section">
  <h2 style="color:#1565c0">Detail Hasil Per Dokumen</h2>
  <input type="text" id="searchBox" placeholder="Cari nama file, nama, NIK..."
         oninput="filterRows()">
  <div style="overflow-x:auto">
  <table id="docTable">
    <thead><tr>
      <th>File Asli</th><th>File Baru</th><th>Jenis</th><th>Status</th>
      <th>Nama / Kepala KK</th><th>NIK / No. KK</th><th>Field Compl.</th>
      <th>Confidence</th><th>Waktu</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>

{f1_section}
{waktu_section}
{naming_section}

<footer>Dibuat oleh Sistem Otomatisasi Penamaan dan Pengarsipan Dokumen Identitas Digital &mdash; OCR Multi-Strategi + NLP Rule-Based v{VERSION}</footer>
</div>
<script>
function filterRows(){{
  var q=document.getElementById("searchBox").value.toLowerCase();
  document.querySelectorAll("#docTable tbody tr").forEach(function(r){{
    r.style.display=r.textContent.toLowerCase().includes(q)?"":"none";
  }});
}}
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    log.info(f"  Laporan HTML : {html_path}")
    return html_path



def export_metadata(
    records     : list,
    output_root : str,
    fmt         : str = "both",
    timestamp   : str = None,
) -> None:
    """
    Menyimpan semua metadata hasil ekstraksi OCR ke JSON dan/atau CSV.

    Args:
        records     (list) : List of dict, satu dict per dokumen.
        output_root (str)  : Folder root untuk subfolder metadata/.
        fmt         (str)  : Format output — 'json', 'csv', atau 'both'.
        timestamp   (str)  : Timestamp untuk nama file. Dibuat otomatis
                             jika tidak diberikan.
    """
    if not records:
        log.warning("Tidak ada record untuk diekspor.")
        return

    meta_dir = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt in ("json", "both"):
        _export_json(records, meta_dir, timestamp)

    if fmt in ("csv", "both"):
        _export_csv(records, meta_dir, timestamp)


def _export_json(records: list, meta_dir: str, timestamp: str) -> None:
    """
    Ekspor metadata ke:
      1. File ringkasan tunggal (summary_<ts>.json) — semua dokumen
      2. File individual (individual/<filename>.json) — satu per dokumen

    File individual berguna untuk inspeksi cepat per dokumen.
    """
    # Ringkasan semua dokumen
    summary_path = os.path.join(meta_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info(f"Metadata JSON: {summary_path}")

    # File individual
    indiv_dir = os.path.join(meta_dir, "individual")
    os.makedirs(indiv_dir, exist_ok=True)
    for rec in records:
        raw_slug  = rec.get("nama_file_baru") or rec.get("nama_file_asli", "unknown")
        slug      = re.sub(r"[^\w\-.]", "_", raw_slug)
        fpath     = os.path.join(indiv_dir, f"{slug}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


def _export_csv(records: list, meta_dir: str, timestamp: str) -> None:
    """
    Ekspor metadata ke satu file CSV flat.

    Menggunakan utf-8-sig (BOM) agar Excel otomatis membaca karakter
    Indonesia (huruf seperti á, é, dll) dengan benar di Windows.
    """
    csv_path = os.path.join(meta_dir, f"hasil_ocr_{timestamp}.csv")
    columns  = [
        "nama_file_asli", "nama_file_baru", "subfolder", "jenis_dokumen",
        "status", "nama", "nik", "tempat_lahir", "tanggal_lahir",
        "jenis_kelamin", "alamat",
        # Kolom khusus KK (v5.5)
        "nomor_kk", "nama_kepala", "desa_kelurahan",
        "kecamatan", "kabupaten_kota", "provinsi", "rtrw",
        # Kolom khusus SIM (v5.7)
        "no_sim",
        "field_completeness",
        "fields_filled", "fields_total", "waktu_proses",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info(f"Metadata CSV : {csv_path}")


# =============================================================================
# MODULE 9A — EVALUASI STATISTIK SISTEM
#
# Menghitung metrik evaluasi keseluruhan dari semua dokumen yang diproses.
# Hasil digunakan untuk:
#   1. Print_evaluation() — tampilan box di terminal
#   2. _save_metrics()    — disimpan ke metadata/evaluasi_<ts>.json
#   3. Tabel hasil dalam laporan penelitian
# =============================================================================
def evaluate_results(
    records          : list,
    batch_start_time : float = None,
) -> dict:
    """
    Menghitung metrik evaluasi lengkap dari semua dokumen yang diproses.

    Metrik yang dihitung:
      - Total, valid, error dokumen
      - Success rate (%)
      - Distribusi per jenis dokumen
      - Rata-rata field completeness (semua + per jenis)
      - Statistik waktu: avg, min, max per dokumen

    Args:
        records          (list) : List of dict hasil pemrosesan.
        batch_start_time (float): time.time() saat batch dimulai.

    Returns:
        dict: Semua metrik evaluasi sistem.
    """
    total = len(records)
    if total == 0:
        return {}

    valid = [r for r in records if "VALID" in r.get("status", "")]
    error = [r for r in records if "ERROR" in r.get("status", "")]

    # Distribusi per jenis dokumen
    dist = {}
    for r in records:
        jenis       = r.get("jenis_dokumen", "unknown")
        dist[jenis] = dist.get(jenis, 0) + 1

    # Field completeness per jenis (hanya dari VALID)
    completeness_per_type = {}
    for jenis in ["ktp", "kk", "sim"]:
        subset = [r for r in valid if r.get("jenis_dokumen") == jenis]
        if subset:
            avg = sum(r.get("field_completeness", 0) for r in subset) / len(subset)
            completeness_per_type[jenis] = round(avg, 3)

    all_c = [r.get("field_completeness", 0) for r in valid]
    avg_c = round(sum(all_c) / len(all_c), 3) if all_c else 0.0

    return {
        "total_dokumen"           : total,
        "dokumen_valid"           : len(valid),
        "dokumen_error"           : len(error),
        "success_rate_pct"        : round(len(valid) / total * 100, 1),
        "distribusi_jenis"        : dist,
        "avg_field_completeness"  : avg_c,
        "completeness_per_jenis"  : completeness_per_type,
    }


def print_evaluation(metrics: dict, logger: logging.Logger):
    """
    Mencetak ringkasan statistik sistem dalam format box ke terminal.

    Contoh output:
      ┌─────────────────────────────────────────────────────┐
      │           STATISTIK SISTEM — HASIL PROSES           │
      ├─────────────────────────────────────────────────────┤
      │  Total dokumen diproses    :     50                 │
      │  KTP                       :     25                 │
      │  KK                        :      7                 │
      │  SIM                       :     15                 │
      │  Valid                     :     47                 │
      │  Error                     :      3                 │
      ├─────────────────────────────────────────────────────┤
      │  OCR Success Rate          :  94.0%                 │
      │  Rata-rata field compl.    :  87.5%                 │
      ├─────────────────────────────────────────────────────┤
      │  Field completeness per jenis:                      │
      │  KTP   [████████████████░░░░]  80.0%                │
      │  KK    [████████████████████] 100.0%                │
      │  SIM   [████████████████████] 100.0%                │
      └─────────────────────────────────────────────────────┘

    Args:
        metrics (dict)           : Output dari evaluate_results().
        logger  (logging.Logger) : Logger untuk output.
    """
    if not metrics:
        return

    total    = metrics.get("total_dokumen", 0)
    valid    = metrics.get("dokumen_valid", 0)
    error    = metrics.get("dokumen_error", 0)
    rate     = metrics.get("success_rate_pct", 0)
    compl    = metrics.get("avg_field_completeness", 0) * 100
    dist     = metrics.get("distribusi_jenis", {})
    compl_per = metrics.get("completeness_per_jenis", {})

    W = 55  # lebar kotak (karakter)

    def row(label, value):
        content = f"  {label:<26}: {str(value):>6}"
        return f"{'|'}{content}{' ' * (W - len(content) - 2)}{'|'}"

    def divider():
        return "+" + "-" * (W - 2) + "+"

    def title(text):
        pad   = W - 2 - len(text)
        pad_l = pad // 2
        pad_r = pad - pad_l
        return "|" + " " * pad_l + text + " " * pad_r + "|"

    def bar_row(label, val):
        filled = int(val / 100 * 20)
        try:
            # Coba Unicode block characters (tampilan lebih rapi)
            bar = chr(9608) * filled + chr(9617) * (20 - filled)
            bar.encode(sys.stdout.encoding or 'utf-8', errors='strict')
        except (UnicodeEncodeError, LookupError):
            # Fallback ASCII untuk terminal yang tidak mendukung Unicode
            bar = "#" * filled + "." * (20 - filled)
        content = f"  {label.upper():<5} [{bar}] {val:5.1f}%"
        return f"{'|'}{content}{' ' * (W - len(content) - 2)}{'|'}"

    lines = [
        "+" + "-" * (W - 2) + "+",
        title("STATISTIK SISTEM - HASIL PROSES"),
        divider(),
        row("Total dokumen diproses", total),
    ]
    for jenis in ["ktp", "kk", "sim", "unknown"]:
        if dist.get(jenis, 0) > 0:
            lines.append(row(jenis.upper(), dist[jenis]))
    lines += [
        row("Valid", valid),
        row("Error", error),
        divider(),
        row("OCR Success Rate", f"{rate:.1f}%"),
        row("Rata-rata field compl.", f"{compl:.1f}%"),
    ]
    if compl_per:
        lines.append(divider())
        pad = W - 2 - len("  Field completeness per jenis:")
        lines.append(f"|  Field completeness per jenis:{' ' * pad}|")
        for jenis in ["ktp", "kk", "sim"]:
            if jenis in compl_per:
                lines.append(bar_row(jenis, compl_per[jenis] * 100))
    lines.append("+" + "-" * (W - 2) + "+")

    logger.info("")
    for ln in lines:
        logger.info(ln)
    logger.info("")


# =============================================================================
# MODULE 9B — EVALUASI KONSISTENSI PENAMAAN FILE
#
# Memverifikasi apakah setiap file output sudah sesuai dengan naming convention:
#   Format  : <jenisdokumen>_<nama_pemilik>.<ext>
#   Jenis   : hanya 'ktp', 'kk', atau 'sim' (lowercase)
#   Nama    : lowercase, underscore sebagai pemisah, hanya alfanumerik
#   Ekstensi: salah satu dari SUPPORTED_EXT
#   Aturan  : tidak ada spasi, tidak ada huruf kapital
#
# Menghasilkan:
#   - compliance_pct   : persentase file sesuai convention
#   - detail_invalid   : daftar file yang tidak sesuai + alasan
#   - File laporan     : metadata/naming_report_<ts>.json
#                        metadata/naming_violations_<ts>.csv
#
# Berguna untuk membuktikan di laporan bahwa sistem menghasilkan penamaan
# yang konsisten dan dapat direproduksi.
# =============================================================================
def evaluate_naming_convention(records: list) -> dict:
    """
    Memeriksa konsistensi penamaan semua file VALID terhadap naming convention.

    Aturan yang diperiksa (sesuai proposal):
      1. Prefix harus 'ktp_', 'kk_', atau 'sim_'
      2. Nama pemilik: lowercase, hanya huruf, angka, underscore
      3. Ekstensi: .jpg, .jpeg, .png, .bmp, .tiff, .tif
      4. Tidak ada spasi, tidak ada huruf kapital
      5. Prefix harus sesuai dengan jenis_dokumen yang terdeteksi

    Args:
        records (list): List of dict hasil pemrosesan.

    Returns:
        dict:
            compliance_pct  (float) : % file sesuai convention
            valid           (int)   : jumlah file sesuai
            invalid         (int)   : jumlah file tidak sesuai
            detail_invalid  (list)  : [{file, issues}, ...] untuk laporan
    """
    valid_records = [r for r in records if "VALID" in r.get("status", "")]
    if not valid_records:
        return {
            "compliance_pct" : 0.0,
            "valid"          : 0,
            "invalid"        : 0,
            "detail_invalid" : [],
        }

    compliant     = 0
    non_compliant = []

    for r in valid_records:
        fname  = r.get("nama_file_baru", "")
        jenis  = r.get("jenis_dokumen", "")
        issues = []

        if not fname:
            issues.append("nama_file_baru kosong")
        else:
            # Aturan 1 & 2 & 3: cek pola keseluruhan
            if not NAMING_PATTERN.match(fname):
                issues.append(
                    f"tidak sesuai pola '{jenis}_<nama_lowercase>.<ext>'"
                )
            # Aturan 5: prefix harus sesuai jenis dokumen
            if jenis and not fname.startswith(f"{jenis}_"):
                actual = fname.split("_")[0]
                issues.append(
                    f"prefix '{actual}' tidak sesuai jenis '{jenis}'"
                )
            # Aturan 4a: tidak ada huruf kapital
            base = os.path.splitext(fname)[0]
            if base != base.lower():
                issues.append("mengandung huruf kapital")
            # Aturan 4b: tidak ada spasi
            if " " in fname:
                issues.append("mengandung spasi")

        if issues:
            non_compliant.append({"file": fname, "issues": issues})
        else:
            compliant += 1

    total = len(valid_records)
    pct   = round(compliant / total * 100, 1) if total else 0.0

    return {
        "compliance_pct" : pct,
        "valid"          : compliant,
        "invalid"        : len(non_compliant),
        "detail_invalid" : non_compliant,
    }


def export_naming_report(
    naming_result : dict,
    output_root   : str,
    timestamp     : str,
) -> None:
    """
    Menyimpan laporan konsistensi penamaan ke JSON dan CSV.

    Output:
      metadata/naming_report_<ts>.json    — ringkasan + detail semua file
      metadata/naming_violations_<ts>.csv — hanya file yang tidak sesuai

    Args:
        naming_result (dict) : Output dari evaluate_naming_convention().
        output_root   (str)  : Folder root.
        timestamp     (str)  : Timestamp untuk nama file.
    """
    meta_dir = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # JSON ringkasan
    json_path = os.path.join(meta_dir, f"naming_report_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(naming_result, f, ensure_ascii=False, indent=2)
    log.info(f"Naming report : {json_path}")

    # CSV violations (hanya jika ada yang tidak sesuai)
    violations = naming_result.get("detail_invalid", [])
    if violations:
        csv_path = os.path.join(
            meta_dir, f"naming_violations_{timestamp}.csv"
        )
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "issues"])
            writer.writeheader()
            for item in violations:
                writer.writerow({
                    "file"  : item["file"],
                    "issues": "; ".join(item["issues"]),
                })
        log.info(f"Naming violations: {csv_path}")


# =============================================================================
# MODULE 9C — EVALUASI PRECISION / RECALL / F1-SCORE
#
# Membandingkan hasil ekstraksi OCR dengan ground truth untuk mengukur
# akurasi per field secara kuantitatif.
#
# Ground truth: file CSV dengan kolom:
#   nama_file, nama, nik, tempat_lahir, tanggal_lahir, jenis_kelamin,
#   jenis_dokumen
#   Nilai kosong ("") = field tidak dievaluasi untuk dokumen tersebut.
#
# Definisi TP/FP/FN untuk ekstraksi teks:
#   TP = field diekstrak DAN nilainya cocok dengan ground truth
#        (setelah normalisasi: lowercase, hapus tanda baca)
#   FP = field diekstrak TAPI nilainya SALAH
#   FN = field TIDAK diekstrak (kosong) padahal ada di ground truth
#
# Rumus (Rusli et al., 2020, Sec. IV — eq. 5, 6, 7):
#   Precision = TP / (TP + FP)
#   Recall    = TP / (TP + FN)
#   F1-Score  = 2 * (P * R) / (P + R)
#
# Macro average = rata-rata sederhana semua field yang dievaluasi.
# =============================================================================
def load_ground_truth(csv_path: str) -> dict:
    """
    Membaca file ground truth CSV dan mengindeksnya berdasarkan nama file.

    Format CSV (header wajib ada, nilai boleh kosong):
      nama_file, nama, nik, tempat_lahir, tanggal_lahir,
      jenis_kelamin, jenis_dokumen

    Contoh baris:
      ktp_dipca.jpg, Dipca Anugrah, 3216082501030001, Bekasi, 25-01-2003,
      LAKI-LAKI, ktp

    Args:
        csv_path (str): Path ke file CSV ground truth.

    Returns:
        dict: {nama_file: {field: nilai, ...}, ...}

    Raises:
        FileNotFoundError : Jika file tidak ditemukan.
        ValueError        : Jika kolom wajib 'nama_file' tidak ada.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] File ground truth tidak ditemukan: {csv_path}\n"
            f"  Buat file CSV dengan kolom:\n"
            f"    nama_file, nama, nik, tempat_lahir, tanggal_lahir,\n"
            f"    jenis_kelamin, jenis_dokumen\n"
            f"  Jalankan --demo untuk melihat contoh format."
        )

    gt = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader  = csv.DictReader(f)
        cols    = reader.fieldnames or []
        if "nama_file" not in cols:
            raise ValueError(
                f"\n[ERROR] Kolom wajib 'nama_file' tidak ada di ground truth.\n"
                f"  Kolom yang ditemukan: {cols}\n"
            )
        for row in reader:
            fname = row.get("nama_file", "").strip()
            if fname:
                entry = {}
                for k, v in row.items():
                    if not k:
                        continue
                    k = k.strip().lower()
                    # Penanganan alias kolom KK
                    if k == "kepala_keluarga": k = "nama_kepala"
                    elif k == "rt_rw": k = "rtrw"
                    
                    if k != "nama_file":
                        entry[k] = v.strip()
                gt[fname] = entry

    log.info(f"Ground truth dimuat: {len(gt)} entri dari {csv_path}")
    return gt


def _normalize_for_compare(val: str) -> str:
    """
    Normalisasi nilai untuk perbandingan yang toleran terhadap format.

    Transformasi:
      - Lowercase
      - Normalisasi separator tanggal (-, / dianggap sama)
      - Hapus karakter selain huruf, angka, dash, spasi
      - Normalisasi whitespace

    Contoh:
      "Sri Rejeki" → "sri rejeki"
      "25/01/2003" → "25-01-2003"
      "LAKI-LAKI"  → "laki-laki"
    """
    if not val:
        return ""
    v = val.lower().strip()
    v = re.sub(r"[/]", "-", v)
    v = re.sub(r"[^a-z0-9\- ]", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def calculate_f1_score(records: list, ground_truth: dict) -> dict:
    """
    Menghitung Precision, Recall, F1-Score per field dan macro average.

    Algoritma:
      1. Buat index records berdasarkan nama_file_asli
      2. Untuk setiap dokumen di ground truth:
         a. Cari record yang cocok berdasarkan nama file
         b. Untuk setiap field (non-kosong di GT):
            - Bandingkan nilai ekstraksi vs GT (setelah normalisasi)
            - Increment TP / FP / FN sesuai hasil
      3. Hitung P, R, F1 per field
      4. Hitung macro average

    Args:
        records      (list) : Output dari process_folder() (list of dict).
        ground_truth (dict) : Output dari load_ground_truth().

    Returns:
        dict:
            per_field          : {field: {tp, fp, fn, precision, recall, f1_score}}
            macro_precision    : float
            macro_recall       : float
            macro_f1_score     : float
            dokumen_dievaluasi : int
            dokumen_tidak_tercocok: int
            total_ground_truth : int
    """
    eval_fields = [
        "nama", "nik", "tempat_lahir", "tanggal_lahir",
        "jenis_kelamin", "jenis_dokumen", "alamat",
        "nomor_kk", "nama_kepala", "desa_kelurahan",
        "kecamatan", "kabupaten_kota", "provinsi", "rtrw",
        "no_sim",
    ]
    counters = {f: {"tp": 0, "fp": 0, "fn": 0} for f in eval_fields}

    # Index records untuk pencarian O(1).
    # Dua index dibangun karena ground truth bisa mencatat:
    #   - nama_file_asli  : nama file sebelum diproses (mode normal)
    #   - nama_file_baru  : nama file setelah rename / preview dry-run
    rec_index_asli = {r.get("nama_file_asli", ""): r for r in records}
    rec_index_baru = {r.get("nama_file_baru", ""): r for r in records}
    matched_docs = 0
    unmatched_docs = 0

    for fname, gt_fields in ground_truth.items():
        rec = rec_index_asli.get(fname) or rec_index_baru.get(fname)

        if rec is None:
            # Dokumen ada di ground truth tapi tidak ter-match ke record manapun.
            # Ini terjadi ketika: (1) OCR salah baca nama sehingga nama_file_baru
            # berbeda, atau (2) dokumen diklasifikasi UNKNOWN dan masuk ERROR/.
            # Secara metodologis, semua field non-kosong di GT dihitung sebagai FN
            # karena sistem gagal mengidentifikasi / mengekstrak dokumen tersebut.
            unmatched_docs += 1
            for field in eval_fields:
                gt_val = _normalize_for_compare(gt_fields.get(field, ""))
                if gt_val:
                    counters[field]["fn"] += 1  # sistem gagal total pada dokumen ini
            continue

        matched_docs += 1

        for field in eval_fields:
            gt_val  = _normalize_for_compare(gt_fields.get(field, ""))
            ocr_val = _normalize_for_compare(rec.get(field, ""))

            if not gt_val:
                continue  # field kosong di GT = tidak dievaluasi

            if ocr_val and ocr_val == gt_val:
                counters[field]["tp"] += 1      # benar
            elif ocr_val and ocr_val != gt_val:
                counters[field]["fp"] += 1      # salah isi
            else:
                counters[field]["fn"] += 1      # tidak terekstrak

    # Hitung metrik per field
    results         = {}
    all_p, all_r, all_f1 = [], [], []

    for field, c in counters.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        results[field] = {
            "tp"        : tp,
            "fp"        : fp,
            "fn"        : fn,
            "precision" : round(precision, 3),
            "recall"    : round(recall, 3),
            "f1_score"  : round(f1, 3),
        }

        if (tp + fp + fn) > 0:
            all_p.append(precision)
            all_r.append(recall)
            all_f1.append(f1)

    macro_p  = round(sum(all_p)  / len(all_p),  3) if all_p  else 0.0
    macro_r  = round(sum(all_r)  / len(all_r),  3) if all_r  else 0.0
    macro_f1 = round(sum(all_f1) / len(all_f1), 3) if all_f1 else 0.0

    return {
        "per_field"             : results,
        "macro_precision"       : macro_p,
        "macro_recall"          : macro_r,
        "macro_f1_score"        : macro_f1,
        "dokumen_dievaluasi"    : matched_docs,
        "dokumen_tidak_tercocok": unmatched_docs,
        "total_ground_truth"    : len(ground_truth),
    }


def print_f1_report(f1_result: dict, logger: logging.Logger) -> None:
    """
    Mencetak tabel Precision / Recall / F1-Score ke terminal.

    Format tabel (cocok untuk disalin ke laporan penelitian):
      +------------------+------+------+------+----+----+----+
      | Field            |  P   |  R   |  F1  | TP | FP | FN |
      +------------------+------+------+------+----+----+----+
      | Nama             | 0.92 | 0.88 | 0.90 | 22 |  2 |  3 |
      | NIK              | 0.84 | 0.80 | 0.82 | 20 |  4 |  5 |
      | ...              | ...  | ...  | ...  | .. | .. | .. |
      +------------------+------+------+------+----+----+----+
      | MACRO AVERAGE    | 0.88 | 0.85 | 0.87 |    |    |    |
      +------------------+------+------+------+----+----+----+

    Args:
        f1_result (dict)         : Output dari calculate_f1_score().
        logger    (logging.Logger): Logger instance.
    """
    if not f1_result:
        return

    per_field = f1_result.get("per_field", {})
    sep = "+------------------+------+------+------+----+----+----+"

    logger.info("")
    logger.info(sep)
    logger.info(
        f"| {'EVALUASI PRECISION / RECALL / F1-SCORE':<52}|"
    )
    logger.info(
        f"|  Dokumen dievaluasi: "
        f"{f1_result['dokumen_dievaluasi']}/{f1_result['total_ground_truth']}"
        f" ground truth{' ':<28}|"
    )
    unmatched = f1_result.get('dokumen_tidak_tercocok', 0)
    if unmatched > 0:
        logger.info(
            f"|  Tidak ter-match (dihitung sbg FN): {unmatched} dok"
            f"{' ':<16}|"
        )
    logger.info(sep)
    logger.info(
        "| {:<16} | {:^4} | {:^4} | {:^4} | {:^2} | {:^2} | {:^2} |".format(
            "Field", "P", "R", "F1", "TP", "FP", "FN"
        )
    )
    logger.info(sep)

    field_labels = {
        "nama"          : "Nama",
        "nik"           : "NIK",
        "tempat_lahir"  : "Tempat Lahir",
        "tanggal_lahir" : "Tanggal Lahir",
        "jenis_kelamin" : "Jenis Kelamin",
        "jenis_dokumen" : "Jenis Dokumen",
        "alamat"        : "Alamat",
        "nomor_kk"      : "Nomor KK",
        "nama_kepala"   : "Nama Kepala",
        "desa_kelurahan": "Desa/Kelurahan",
        "kecamatan"     : "Kecamatan",
        "kabupaten_kota": "Kab/Kota",
        "provinsi"      : "Provinsi",
        "rtrw"          : "RT/RW",
        "no_sim"        : "No SIM",
    }
    for field, label in field_labels.items():
        m = per_field.get(field, {})
        if not m or (m["tp"] + m["fp"] + m["fn"]) == 0:
            continue
        logger.info(
            "| {:<16} | {:>4.2f} | {:>4.2f} | {:>4.2f} | {:>2} | {:>2} | {:>2} |".format(
                label[:16],
                m["precision"], m["recall"], m["f1_score"],
                m["tp"], m["fp"], m["fn"],
            )
        )

    logger.info(sep)
    logger.info(
        "| {:<16} | {:>4.2f} | {:>4.2f} | {:>4.2f} | {:^2} | {:^2} | {:^2} |".format(
            "MACRO AVG",
            f1_result["macro_precision"],
            f1_result["macro_recall"],
            f1_result["macro_f1_score"],
            "-", "-", "-",
        )
    )
    logger.info(sep)
    logger.info("")


def export_f1_report(f1_result: dict, output_root: str, timestamp: str) -> None:
    """
    Menyimpan hasil evaluasi F1-Score ke JSON dan CSV.

    Output:
      metadata/f1_score_<ts>.json  — hasil lengkap per field
      metadata/f1_score_<ts>.csv   — tabel flat untuk laporan penelitian

    Args:
        f1_result   (dict): Output dari calculate_f1_score().
        output_root (str) : Folder root.
        timestamp   (str) : Timestamp untuk nama file.
    """
    meta_dir = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(meta_dir, f"f1_score_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(f1_result, f, ensure_ascii=False, indent=2)
    log.info(f"F1 JSON: {json_path}")

    # CSV per field (tabel siap pakai untuk laporan)
    csv_path = os.path.join(meta_dir, f"f1_score_{timestamp}.csv")
    field_labels = {
        "nama"          : "Nama",
        "nik"           : "NIK",
        "tempat_lahir"  : "Tempat Lahir",
        "tanggal_lahir" : "Tanggal Lahir",
        "jenis_kelamin" : "Jenis Kelamin",
        "jenis_dokumen" : "Jenis Dokumen",
        "alamat"        : "Alamat",
        "nomor_kk"      : "Nomor KK",
        "nama_kepala"   : "Nama Kepala",
        "desa_kelurahan": "Desa/Kelurahan",
        "kecamatan"     : "Kecamatan",
        "kabupaten_kota": "Kab/Kota",
        "provinsi"      : "Provinsi",
        "rtrw"          : "RT/RW",
        "no_sim"        : "No SIM",
    }
    rows = []
    for field, label in field_labels.items():
        m = f1_result.get("per_field", {}).get(field, {})
        if m and (m["tp"] + m["fp"] + m["fn"]) > 0:
            rows.append({
                "field"     : label,
                "precision" : m["precision"],
                "recall"    : m["recall"],
                "f1_score"  : m["f1_score"],
                "tp"        : m["tp"],
                "fp"        : m["fp"],
                "fn"        : m["fn"],
            })
    rows.append({
        "field"     : "MACRO AVERAGE",
        "precision" : f1_result["macro_precision"],
        "recall"    : f1_result["macro_recall"],
        "f1_score"  : f1_result["macro_f1_score"],
        "tp": "", "fp": "", "fn": "",
    })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["field","precision","recall","f1_score","tp","fp","fn"]
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"F1 CSV : {csv_path}")


# =============================================================================
# MODULE 9D — EVALUASI EFISIENSI WAKTU PROSES
#
# Mengukur dan merangkum performa waktu pemrosesan:
#   - Waktu per dokumen (ms): dicatat dalam _process_single()
#   - Statistik: rata-rata, minimum, maksimum
#   - Total waktu batch
#   - Perbandingan dengan estimasi waktu manual (simulasi)
#
# Estimasi waktu manual didasarkan pada asumsi operator manusia:
#   KTP : 45 detik (baca + ketik 6 field: nama, NIK, TTL, JK, alamat, berlaku)
#   KK  : 60 detik (lebih banyak field anggota keluarga)
#   SIM : 30 detik (field lebih sedikit: nama, TTL, golongan)
#
# Rasio kecepatan = estimasi_manual_total / waktu_batch_sistem
# (Rusli et al., 2020, Sec. IV — pengukuran waktu 4510ms/dokumen)
# =============================================================================
def evaluate_time_efficiency(
    records          : list,
    batch_start_time : float,
) -> dict:
    """
    Menghitung metrik efisiensi waktu proses vs estimasi manual.

    Args:
        records          (list) : List of dict, waktu_proses dalam format "NNN ms".
        batch_start_time (float): time.time() saat batch dimulai.

    Returns:
        dict:
            avg_ms                : rata-rata waktu per dokumen (ms)
            min_ms / max_ms       : waktu tercepat/terlama (ms)
            total_batch_ms        : total waktu batch (ms)
            total_batch_detik     : total waktu batch (detik)
            estimasi_manual_detik : total jika dikerjakan manual (detik)
            rasio_kecepatan       : sistem X kali lebih cepat dari manual
            avg_ms_per_jenis      : rata-rata per jenis dokumen
    """
    durations_ms = []
    per_type_ms  = {}

    for r in records:
        raw = r.get("waktu_proses", "")
        m   = re.search(r"(\d+)", str(raw))
        if m:
            ms    = int(m.group(1))
            jenis = r.get("jenis_dokumen", "unknown")
            durations_ms.append(ms)
            per_type_ms.setdefault(jenis, []).append(ms)

    total_batch_ms = int((time.time() - batch_start_time) * 1000)

    # Estimasi waktu manual total
    manual_total_s = sum(
        MANUAL_TIME_SECONDS.get(r.get("jenis_dokumen", "unknown"), 40)
        for r in records
    )
    system_s = total_batch_ms / 1000
    speedup  = round(manual_total_s / system_s, 1) if system_s > 0 else 0

    avg_per_type = {
        jenis: round(sum(times) / len(times))
        for jenis, times in per_type_ms.items() if times
    }

    return {
        "avg_ms"                : round(sum(durations_ms) / len(durations_ms)) if durations_ms else 0,
        "min_ms"                : min(durations_ms) if durations_ms else 0,
        "max_ms"                : max(durations_ms) if durations_ms else 0,
        "total_batch_ms"        : total_batch_ms,
        "total_batch_detik"     : round(system_s, 1),
        "estimasi_manual_detik" : manual_total_s,
        "rasio_kecepatan"       : speedup,
        "avg_ms_per_jenis"      : avg_per_type,
    }


def print_time_report(time_result: dict, logger: logging.Logger) -> None:
    """
    Mencetak ringkasan efisiensi waktu proses ke terminal.

    Contoh output:
      +-----------------------------------------------------+
      |         EVALUASI EFISIENSI WAKTU PROSES             |
      +-----------------------------------------------------+
      |  Rata-rata per dokumen       :   4510 ms            |
      |  Tercepat                    :   2100 ms            |
      |  Terlama                     :   8200 ms            |
      +-----------------------------------------------------+
      |  Total waktu batch           :    225.5 dtk         |
      |  Estimasi manual             :   2250.0 dtk         |
      |  Sistem lebih cepat          :      10.0x           |
      +-----------------------------------------------------+

    Args:
        time_result (dict)          : Output dari evaluate_time_efficiency().
        logger      (logging.Logger): Logger instance.
    """
    if not time_result:
        return

    W   = 55
    sep = "+" + "-" * (W - 2) + "+"

    def row(label, value):
        content = f"  {label:<30}: {str(value):>8}"
        return f"|{content}{' ' * (W - len(content) - 2)}|"

    def title(text):
        pad   = W - 2 - len(text)
        pad_l = pad // 2
        pad_r = pad - pad_l
        return "|" + " " * pad_l + text + " " * pad_r + "|"

    lines = [
        sep,
        title("EVALUASI EFISIENSI WAKTU PROSES"),
        sep,
        row("Rata-rata per dokumen",  f"{time_result['avg_ms']} ms"),
        row("Tercepat",               f"{time_result['min_ms']} ms"),
        row("Terlama",                f"{time_result['max_ms']} ms"),
        sep,
        row("Total waktu batch",      f"{time_result['total_batch_detik']} dtk"),
        row("Estimasi waktu manual",  f"{time_result['estimasi_manual_detik']} dtk"),
        row("Sistem lebih cepat",     f"{time_result['rasio_kecepatan']}x"),
    ]
    if time_result.get("avg_ms_per_jenis"):
        lines.append(sep)
        lines.append(title("Rata-rata waktu per jenis (ms)"))
        for jenis, avg in time_result["avg_ms_per_jenis"].items():
            lines.append(row(f"  {jenis.upper()}", f"{avg} ms"))
    lines.append(sep)

    logger.info("")
    for ln in lines:
        logger.info(ln)
    logger.info("")


def _save_metrics(metrics: dict, root_input: str, timestamp: str = None) -> None:
    """
    Simpan semua metrik evaluasi ke JSON di folder metadata/.

    File ini berisi statistik lengkap satu sesi pemrosesan dan berguna
    untuk perbandingan antar sesi dalam penelitian longitudinal.

    Args:
        metrics     (dict): Output dari evaluate_results() + tambahan.
        root_input  (str) : Folder root.
        timestamp   (str) : Timestamp. Dibuat otomatis jika None.
    """
    meta_dir = os.path.join(root_input, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = os.path.join(meta_dir, f"evaluasi_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info(f"Metrik evaluasi: {path}")

