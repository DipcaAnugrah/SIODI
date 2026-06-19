"""
config.py — Konfigurasi dan Konstanta Sistem
============================================
Semua konstanta, parameter, template penamaan, dan konfigurasi file.
Tidak mengimpor modul sistem lainnya — diimpor oleh semua modul lain.
"""
import re, os, json, logging
from datetime import datetime

VERSION = "5.7"

# ── Ekstensi & folder ──────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SKIP_DIRS     = {"KTP", "KK", "SIM", "ERROR", "logs", "metadata"}
MANUAL_TIME_SECONDS = {"ktp": 45, "kk": 60, "sim": 30, "unknown": 40}
NAMING_PATTERN = re.compile(
    r"^(ktp|kk|sim)_[a-z][a-z0-9_]{1,99}\.(jpg|jpeg|png|bmp|tiff|tif)$"
)

NAME_STOPWORDS = {
    "laki","perempuan","islam","kristen","hindu","budha","katolik","protestan","konghucu",
    "wni","wna","kawin","belum","cerai","pelajar","mahasiswa","pegawai","swasta","buruh",
    "petani","wiraswasta","provinsi","kabupaten","kota","kecamatan","kelurahan","desa",
    "republik","indonesia","daerah","istimewa","nik","nama","alamat","agama","pekerjaan",
    "golongan","darah","berlaku","hingga","hidup","seumur","status","perkawinan",
    "kewarganegaraan","tempat","lahir","tanggal","jenis","kelamin","rtrw","rt","rw",
    "kel","des","kec","no","nomor","kepala","keluarga","hubungan","dalam","pencatatan",
    "sipil","kependudukan","dinas","dikeluarkan","yogyakarta","jakarta","bandung",
    "surabaya","medan","bekasi","depok","tangerang","bogor","bali","badung","denpasar",
    "sleman","bantul","klaten","magelang","semarang","solo","malang","palembang",
    "makassar","balikpapan","samarinda","manado","pontianak","banjarmasin","pekanbaru",
    "jambi","kp","rawa","banteng","barat","timur","utara","selatan","jl","jalan","gg",
    "gang","lr","lorong","blok","bl","perum","perumahan","griya","komplek","kav",
    "kavling","ckr","cikaret","sei","sungai","komp","metro","jaya","polda","polri",
    "driving","licence","license","surat","izin","mengemudi","sim","kab","mhs",
}

WORD_TO_NUM = {
    "L":"1","l":"1","I":"1","i":"1","O":"0","o":"0","D":"0",
    "Z":"2","z":"2","S":"5","s":"5","b":"6","G":"6","B":"8","?":"7","A":"4",
}

# ── Batas validasi ──────────────────────────────────────────────────────────────
MAX_NAME_WORDS=4; MIN_NAME_WORDS=1; MIN_WORD_LEN=3; MIN_NAME_LEN=5; MIN_VOWEL_RATIO=0.20
MAX_NOISE_RATIO=0.40; MIN_NIK_DIGITS=15; MIN_KK_DIGITS=15
PORTRAIT_THRESHOLD=1.1; SQUARE_THRESHOLD=0.85; ROT180_SCORE_MARGIN=1.2
MAX_ADDR_LEN=200; MAX_ADDR_LINES=3; MIN_IMG_DIMENSION=800; UPSCALE_TARGET=1600

# ── Template ─────────────────────────────────────────────────────────────────────
DEFAULT_FILE_TEMPLATE   = "{jenis}_{nama}"
DEFAULT_FOLDER_TEMPLATE = "{JENIS}/{jenis}_{nama}"
DEFAULT_SEPARATOR       = "_"
VALID_TEMPLATE_VARS = frozenset({
    "jenis","JENIS","nama","NAMA","nik","nik6","tgl","tgl_compact",
    "tempat","jk","tanggal","timestamp",
    "nokk","nokk6","kepala","KEPALA","desa","kecamatan","kabupaten","provinsi","rtrw",
})

CONFIG_DEFAULTS = {
    "input":"Dokumen","lang":"ind","export":"both",
    "file_template":DEFAULT_FILE_TEMPLATE,"folder_template":DEFAULT_FOLDER_TEMPLATE,
    "separator":DEFAULT_SEPARATOR,"deskew":True,"dry_run":False,
    "ground_truth":None,"tesseract_path":None,
}

log = logging.getLogger("otomatisasi_dokumen")


def build_template_context(doc_type: str, fields: dict, sep: str = "_") -> dict:
    now = datetime.now()
    def slugify(val):
        v = (val or "").lower().strip()
        v = re.sub(r"[^a-z0-9]+", sep, v)
        return v.strip(sep + "-") or ""

    nama_slug=slugify(fields.get("nama","")) or "unknown"
    tempat_slug=slugify(fields.get("tempat_lahir",""))
    nik_raw=re.sub(r"[^0-9]","",fields.get("nik","") or "")
    tgl_raw=fields.get("tanggal_lahir","") or ""
    tgl_compact=re.sub(r"[^0-9]","",tgl_raw)
    tgl_display=tgl_raw.replace("/","-")
    jk_full=(fields.get("jenis_kelamin","") or "").upper()
    jk={"LAKI-LAKI":"L","PEREMPUAN":"P"}.get(jk_full,"X")

    nokk_raw=kepala_slug=desa_slug=kec_slug=kab_slug=prov_slug=rtrw_slug=""
    if doc_type == "kk":
        nokk_raw=re.sub(r"[^0-9]","",fields.get("nomor_kk","") or "")
        kepala_slug=slugify(fields.get("nama_kepala",""))
        desa_slug=slugify(fields.get("desa_kelurahan",""))
        kec_slug=slugify(fields.get("kecamatan",""))
        kab_slug=slugify(fields.get("kabupaten_kota",""))
        prov_slug=slugify(fields.get("provinsi",""))
        rtrw_raw=fields.get("rtrw","") or ""
        rtrw_slug=rtrw_raw.replace("/",sep) if rtrw_raw else ""

    nik_for_template=nokk_raw if (doc_type=="kk" and nokk_raw) else nik_raw
    return {
        "jenis":doc_type.lower(),"JENIS":doc_type.upper(),
        "nama":nama_slug,"NAMA":nama_slug.upper(),
        "nik":nik_for_template or "nonik",
        "nik6":(nik_for_template[:6] if len(nik_for_template)>=6 else nik_for_template.ljust(6,"0")),
        "tgl":tgl_display or "notgl","tgl_compact":tgl_compact or "notgl",
        "tempat":tempat_slug or "notempat","jk":jk,
        "tanggal":now.strftime("%Y%m%d"),"timestamp":now.strftime("%Y%m%d_%H%M%S"),
        "nokk":nokk_raw or "",
        "nokk6":(nokk_raw[:6] if len(nokk_raw)>=6 else nokk_raw.ljust(6,"0")) if nokk_raw else "",
        "kepala":kepala_slug or "","KEPALA":kepala_slug.upper() if kepala_slug else "",
        "desa":desa_slug or "","kecamatan":kec_slug or "",
        "kabupaten":kab_slug or "","provinsi":prov_slug or "","rtrw":rtrw_slug or "",
    }


def apply_template(template: str, context: dict, sep: str = "_") -> str:
    class SafeDict(dict):
        def __missing__(self, key): return ""
    try:
        result = template.format_map(SafeDict(context))
    except Exception:
        result = "{}_{}".format(context.get("jenis","dok"),context.get("nama","unknown"))
    parts = result.replace("\\", "/").split("/")
    clean = []
    for part in parts:
        safe = re.sub(r"[^a-zA-Z0-9_\-\.]","",part)
        safe = re.sub(r"[_\-]{2,}",sep,safe).strip(sep+"-")
        if safe: clean.append(safe)
    fallback = "{}_{}".format(context.get("jenis","dok"),context.get("nama","unknown"))
    return "/".join(clean) if clean else fallback


def validate_template(template: str) -> tuple:
    if not template or not template.strip():
        return False, "Template tidak boleh kosong."
    found   = re.findall(r"\{(\w+)\}", template)
    unknown = [v for v in found if v not in VALID_TEMPLATE_VARS]
    if unknown:
        return False, ("Variabel tidak dikenal: {{{}}}\n  Jalankan --template-help.").format(
            ", ".join(unknown))
    return True, "OK"


def print_template_help() -> None:
    print("""
==========================================================
  PANDUAN TEMPLATE PENAMAAN KUSTOM  (--template-help)
==========================================================
VARIABEL UMUM:  {jenis} {JENIS} {nama} {NAMA} {nik} {nik6}
                {tgl} {tgl_compact} {tempat} {jk} {tanggal} {timestamp}
VARIABEL KK:    {nokk} {nokk6} {kepala} {KEPALA}
                {desa} {kecamatan} {kabupaten} {provinsi} {rtrw}

CONTOH:
  python main.py --input Dokumen
    -> KTP/ktp_dipca_anugrah/ktp_dipca_anugrah.jpeg

  python main.py --input Dokumen --file-template "{nik}_{nama}"
    -> KTP/ktp_dipca_anugrah/3216082501030001_dipca_anugrah.jpeg

  python main.py --input Dokumen --folder-template "{JENIS}/{kabupaten}/{kecamatan}"
    -> KK/bekasi/cikarang_barat/kk_dadang_suhendang.jpeg
==========================================================
""")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\n[ERROR] File config tidak ditemukan: {config_path}\n"
            f"  Buat template: python main.py --create-config"
        )
    ext = os.path.splitext(config_path)[1].lower()
    if ext == ".json":
        with open(config_path,"r",encoding="utf-8") as fh:
            try: raw = json.load(fh)
            except json.JSONDecodeError as e: raise ValueError(f"\n[ERROR] Config JSON tidak valid: {e}")
    elif ext in (".yaml",".yml"):
        try:
            import yaml
            with open(config_path,"r",encoding="utf-8") as fh: raw = yaml.safe_load(fh)
        except ImportError: raise ValueError("\n[ERROR] Format YAML butuh PyYAML.\n  pip install pyyaml")
        except Exception as e: raise ValueError(f"\n[ERROR] Config YAML tidak valid: {e}")
    else:
        raise ValueError(f"\n[ERROR] Format config tidak dikenal: '{ext}'")

    result = dict(CONFIG_DEFAULTS)
    unknown = []
    for k,v in raw.items():
        if k.startswith("_"): continue
        if k in CONFIG_DEFAULTS: result[k] = v
        else: unknown.append(k)
    if unknown: log.warning(f"Kunci config tidak dikenal diabaikan: {unknown}")
    return result


def create_config_template(output_path: str = "config.json") -> None:
    template = {"_keterangan":{
        "input":"Folder utama berisi subfolder dokumen",
        "lang":"Kode bahasa Tesseract: ind / ind+eng",
        "export":"Format metadata: json / csv / both",
        "file_template":"Template nama file. Var: {jenis},{nama},{nik},...",
        "folder_template":"Template subfolder. Gunakan / untuk bertingkat.",
        "separator":"Pemisah kata: _ atau -",
        "deskew":"true = koreksi kemiringan gambar sebelum OCR",
        "dry_run":"true = preview saja, file TIDAK dipindah",
        "ground_truth":"Path CSV ground truth untuk F1. null = lewati.",
        "tesseract_path":"Path tesseract.exe (Windows). null = dari PATH.",
    }, **{k:v for k,v in CONFIG_DEFAULTS.items()}}
    with open(output_path,"w",encoding="utf-8") as fh:
        json.dump(template,fh,ensure_ascii=False,indent=2)
    log.info(f"Template config dibuat: {output_path}")
