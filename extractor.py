"""
extractor.py — Klasifikasi Dokumen dan Ekstraksi Field
=======================================================
Menangani:
  - Klasifikasi dokumen: KTP / KK / SIM via keyword scoring NLP rule-based (Modul 5)
  - Ekstraksi field KTP: Nama, NIK, Tempat/Tgl Lahir, JK, Alamat (Modul 6)
  - Ekstraksi 8 field KK: No.KK, Kepala KK, Desa, Kec, Kab, Prov, RT/RW, Alamat
  - Ekstraksi 4 field SIM: No.SIM, Nama, Tgl Lahir, Alamat
  - Voting konsensus dari 12 varian OCR per field
  - Confidence score per field (Modul 1E)
  - Validasi dan normalisasi nama orang (_is_plausible_name, _clean_name)
"""
import re, logging
from collections import Counter

from config import (
    NAME_STOPWORDS, WORD_TO_NUM,
    MAX_NAME_WORDS, MIN_NAME_WORDS, MIN_WORD_LEN, MIN_NAME_LEN, MIN_VOWEL_RATIO,
    MAX_NOISE_RATIO, MIN_NIK_DIGITS, MIN_KK_DIGITS,
    MAX_ADDR_LEN, MAX_ADDR_LINES,
)

log = logging.getLogger("otomatisasi_dokumen")

def classify_document(text: str) -> str:
    """
    Mengklasifikasikan teks OCR menjadi jenis dokumen identitas.

    Menggunakan pendekatan scoring berbasis regex. Setiap pola yang cocok
    menambah skor 1. Threshold dan urutan pengecekan menentukan klasifikasi.

    Pola regex dirancang toleran terhadap noise OCR:
      - r"N\\s*[I1]\\s*K" mencocokkan "NIK", "N1K", "N I K"
      - r"POLR[I1]" mencocokkan "POLRI" dan "POLR1"

    Args:
        text (str): Teks OCR yang sudah dibersihkan.

    Returns:
        str: Jenis dokumen — 'ktp', 'kk', 'sim', atau 'unknown'.
    """
    upper = text.upper()

    # ── Prioritas 1: KARTU KELUARGA ───────────────────────────────────────────
    # Kata kunci eksklusif KK — tidak pernah muncul di KTP atau SIM.
    # Satu saja yang cocok sudah cukup untuk klasifikasi KK.
    kk_exclusive = [
        r"KARTU\s+KELUARGA",            # header utama dokumen KK
        r"NAMA\s+KEPALA\s+KELUARGA",    # label unik di KK
        r"KEPALA\s+KELUARGA",           # label ringkas kepala keluarga
        r"NO\s*[\.\:]?\s*KK\b",         # nomor KK
        r"NOMOR\s+KK",
        r"STATUS\s+HUBUNGAN",           # kolom hubungan anggota keluarga
        r"HUBUNGAN\s+DALAM\s+KELUARGA",
        r"HUBUNGAN\s+KELUARGA",
        r"PENCATATAN\s+SIPIL",          # footer KK: Dinas Pencatatan Sipil
        r"DINAS\s+KEPENDUDUKAN",
    ]
    if any(re.search(p, upper) for p in kk_exclusive):
        return "kk"

    # ── Prioritas 2: SIM ──────────────────────────────────────────────────
    sim_patterns = [
        r"SURAT\s+I[ZJ]IN\s+MENGEMU",        # "SURAT IZIN MENGEMUDI" (terpotong atau utuh)
        r"I[ZJ]IN\s+MENGEMU",                # "IZIN MENGEMUDI", "IJIN MENGEMUDI" tanpa SURAT
        r"SURAT\s+I[ZJ]IN",                  # "SURAT IJIN" saja
        r"SURAT\s+[IJ]IN",                   # "SURAT JIN" / "SURAT IN" (OCR drop huruf I)
        r"MENGEMU[N]?[DO][I1]?",             # MENGEMUDI / MENGEMUOI / MENGEMUD1 / MENGEMUNDI
        r"GOLONGAN\s+[ABC]",                 # golongan SIM A/B/C
        r"POLR[I1]",                          # penerbit SIM = POLRI
        r"DRIVING\s+LICEN[SC]E?",            # versi bilingual KTP
        r"DRIVING",                           # minimal DRIVING saja
        r"SIM\s+[ABC]\b",                     # "SIM A" / "SIM B" / "SIM C"
    ]
    sim_score = sum(1 for p in sim_patterns if re.search(p, upper))
    # Threshold 1 sudah cukup untuk SIM karena pola di atas sangat spesifik
    # Periksa dulu sebelum KTP agar SIM dengan keyword KABUPATEN tidak salah classified
    if sim_score >= 2:
        return "sim"

    # ── Prioritas 3: KTP ──────────────────────────────────────────────────
    # Threshold 2 (bukan 1) untuk mengurangi false positive pada dokumen lain
    ktp_patterns = [
        r"\bN\s*[I1]\s*K\b",           # NIK (toleran: N1K, N I K)
        r"PROV[I1]NS[I1]",             # PROVINSI
        r"TANDA\s+PENDUDUK",           # Kartu Tanda Penduduk
        r"KARTU\s+TANDA",
        r"\bKTP\b",
        r"REPUBLIK\s+INDONESIA",
        r"JENIS\s+KELAMIN",
        r"GOL\s*[.\s]*DARAH",
        r"BERLAKU\s+HINGGA",
        r"KECAMATAN",
        r"KELURAHAN",
        r"KABUPATEN",
        r"SEUMUR\s+HIDUP",
    ]
    ktp_score = sum(1 for p in ktp_patterns if re.search(p, upper))
    if ktp_score >= 2:
        return "ktp"

    # Fallback skor rendah
    if sim_score >= 1:
        return "sim"
    if ktp_score >= 1:
        return "ktp"

    return "unknown"


# =============================================================================
# MODULE 6 — EKSTRAKSI FIELD DOKUMEN
#
# Mengekstrak 5 field utama dari teks OCR menggunakan NLP rule-based:
#   1. Nama     — 4 strategi berlapis + voting lintas 12 OCR
#   2. NIK      — regex 16 digit + koreksi karakter noise
#   3. Tempat lahir — split sebelum tanggal dari field "Tempat/Tgl Lahir"
#   4. Tanggal lahir — regex pola DD-MM-YYYY
#   5. Jenis kelamin — hanya dua nilai: LAKI-LAKI / PEREMPUAN
#   6. Alamat   — multi-baris dari field "Alamat :"
#
# Setelah ekstraksi, dihitung Field Completeness Score:
#   completeness = field_terisi / field_relevan_per_jenis
#   Field relevan: KTP=6, KK=3, SIM=2
#
# (Rusli et al., 2020, Sec. III-C — Fig. 6: 3-part NLP field splitting)
# =============================================================================
def extract_fields(
    text: str,
    doc_type: str,
    all_texts: list = None,
) -> dict:
    """
    Mengekstrak field penting dari teks OCR sesuai jenis dokumen .

    KTP: nama, nik, tempat_lahir, tanggal_lahir, jenis_kelamin, alamat
    KK : nomor_kk, nama_kepala, desa_kelurahan, kecamatan, kabupaten_kota,
         provinsi, rtrw, alamat  (8 field)
    SIM: nama, tanggal_lahir

    Field KK berdasarkan struktur header Kartu Keluarga:
      - nomor_kk       : nomor KK 16 digit di baris "No. XXXXXX"
      - nama_kepala    : nama kepala keluarga (di bawah nomor KK)
      - desa_kelurahan : nama desa/kelurahan
      - kecamatan      : nama kecamatan
      - kabupaten_kota : nama kabupaten/kota
      - provinsi       : nama provinsi
      - rtrw           : RT/RW dalam format NNN/NNN
      - alamat         : alamat lengkap (KP/JL/PERUM dst)

    Args:
        text      (str)       : Teks OCR terbaik (sudah dibersihkan).
        doc_type  (str)       : Jenis dokumen — 'ktp', 'kk', 'sim', 'unknown'.
        all_texts (list[str]) : Semua 12 varian OCR (untuk voting nama).

    Returns:
        dict dengan kunci field + metadata:
            _field_completeness, _fields_filled, _fields_total
    """
    all_src = [text] + (all_texts or [])

    if doc_type == "kk":
        # ── Voting nomor KK ──────────────────────────────────────────────────
        kk_candidates = [_extract_nomor_kk(t) for t in all_src]
        kk_candidates = [v for v in kk_candidates if v]
        nomor_kk = Counter(kk_candidates).most_common(1)[0][0] if kk_candidates else ""

        # ── Voting nama kepala ───────────────────────────────────────────────
        nama_candidates = [_extract_nama_kepala_kk(t) for t in all_src]
        nama_candidates = [v for v in nama_candidates if v]
        nama_kepala = Counter(nama_candidates).most_common(1)[0][0] if nama_candidates else ""

        # ── Alamat KK ────────────────────────────────────────────────────────
        alamat_kk = _extract_alamat_kk(text, all_src)

        # ── RT/RW ────────────────────────────────────────────────────────────
        rtrw_candidates = [_extract_rtrw(t) for t in all_src]
        rtrw_candidates = [v for v in rtrw_candidates if v]
        rtrw = Counter(rtrw_candidates).most_common(1)[0][0] if rtrw_candidates else ""

        # ── Field header KK baru (v5.5) ──────────────────────────────────────
        # Desa/Kelurahan, Kecamatan, Kabupaten/Kota, Provinsi
        kk_header = _extract_kk_fields_voting(all_src)

        fields = {
            "nama"              : nama_kepala,
            "nik"               : nomor_kk,
            "tempat_lahir"      : "",
            "tanggal_lahir"     : "",
            "jenis_kelamin"     : "",
            "alamat"            : alamat_kk,
            # Field KK standar (selalu ada)
            "nomor_kk"          : nomor_kk,
            "nama_kepala"       : nama_kepala,
            "rtrw"              : rtrw,
            # Field header KK (desa, kecamatan, kabupaten, provinsi)
            "desa_kelurahan"    : kk_header.get("desa_kelurahan", ""),
            "kecamatan"         : kk_header.get("kecamatan", ""),
            "kabupaten_kota"    : kk_header.get("kabupaten_kota", ""),
            "provinsi"          : kk_header.get("provinsi", ""),
        }
        relevant = [
            "nomor_kk", "nama_kepala", "desa_kelurahan",
            "kecamatan", "kabupaten_kota", "provinsi",
            "rtrw", "alamat",
        ]

    else:
        alamat = _extract_alamat(text)
        # SIM: jika _extract_alamat kosong, coba _extract_alamat_sim
        if doc_type == "sim" and not alamat:
            alamat = _extract_alamat_sim(text, all_src)
        fields = {
            "nama"          : _extract_name(text, doc_type, all_texts),
            "nik"           : _extract_nik_voting(all_src),
            "tempat_lahir"  : _extract_tempat_voting(all_src),
            "tanggal_lahir" : _extract_tanggal_voting(all_src),
            "jenis_kelamin" : _extract_jenis_kelamin(text),
            "alamat"        : alamat,
            # Field khusus SIM
            "no_sim"        : _extract_no_sim_voting(all_src) if doc_type == "sim" else "",
        }
        relevant = {
            "ktp": ["nama", "nik", "tempat_lahir", "tanggal_lahir",
                    "jenis_kelamin", "alamat"],
            "sim": ["no_sim", "nama", "tanggal_lahir", "alamat"],
        }.get(doc_type, ["nama"])

    filled = sum(1 for k in relevant if fields.get(k))
    fields["_field_completeness"] = round(filled / len(relevant), 2)
    fields["_fields_filled"]      = filled
    fields["_fields_total"]       = len(relevant)
    return fields


def _extract_nik_voting(all_src: list) -> str:
    """Voting NIK dari semua varian OCR — pilih yang paling sering muncul."""
    candidates = [_extract_nik(t) for t in all_src]
    candidates = [v for v in candidates if v]
    return Counter(candidates).most_common(1)[0][0] if candidates else ""


def _extract_tempat_voting(all_src: list) -> str:
    """Voting tempat lahir dari semua varian OCR."""
    candidates = [_extract_tempat_lahir(t) for t in all_src]
    candidates = [v for v in candidates if v and len(v) > 2]
    return Counter(candidates).most_common(1)[0][0] if candidates else ""


def _extract_tanggal_voting(all_src: list) -> str:
    """Voting tanggal lahir dari semua varian OCR."""
    candidates = [_extract_tanggal_lahir(t) for t in all_src]
    candidates = [v for v in candidates if v]
    return Counter(candidates).most_common(1)[0][0] if candidates else ""


# ── Field khusus KK ───────────────────────────────────────────────────────────

def _extract_nomor_kk(text: str) -> str:
    """
    Ekstrak nomor KK (16 digit) dari teks OCR Kartu Keluarga.

    Nomor KK di kartu fisik dicetak dengan font SANGAT BESAR di baris atas,
    format: "No. XXXXXXXXXXXXXXXX" atau "NOMOR KK XXXXXXXXXXXXXXXX"

    Bug yang diperbaiki di v5.8.0:
    ─────────────────────────────
    Bug 1 — Pola A terlalu longgar, menangkap teks setelah NIK/tanggal:
      Regex lama `N[o0ua][. -]*` juga cocok dengan "No" dalam kata seperti
      "No.HP", "No.RT", "NOMOR NIK" dan kemudian menangkap NIK 16-digit yang
      ada di bawahnya. Diperbaiki dengan memisahkan pola "NOMOR KK" (eksplisit)
      dari pola "No." (generik) dan menambahkan anchor akhir baris.

    Bug 2 — Pola B (fallback) tidak membedakan NIK vs nomor KK:
      Pola B mencari "sequence 15-18 karakter dimulai 1-6" tanpa konteks.
      NIK (16 digit) juga dimulai 1-6 dan memenuhi kriteria ini, sehingga
      NIK selalu masuk sebagai kandidat dan seringkali memenangkan voting.
      Diperbaiki:
        a. Pola B hanya aktif jika Pola A tidak menemukan kandidat valid.
        b. Tambah filter: kandidat Pola B tidak boleh identik dengan NIK yang
           sudah ditemukan oleh _extract_nik di teks yang sama (cross-check).
        c. Tambah validasi panjang tepat: hanya terima 16 digit (bukan 15+).

    Bug 3 — Spasi antar karakter font besar tidak di-strip sebelum translasi:
      Font besar di KK menyebabkan Tesseract menyisipkan spasi antar digit,
      contoh: "3 5 2 4 1 4 2 0 2 4 1 0 9 8". Pola lama menghapus spasi SETELAH
      translasi. Diperbaiki: strip spasi lebih agresif sebelum dan sesudah
      translasi, dan tambah penanganan pola "digit spasi digit" berulang.

    Perbaikan v5.7.6-KK (tetap dipertahankan):
      - Pola NOMOR KK / No. KK / N0. (digit nol)
      - Translasi karakter OCR noise: O→0, l→1, I→1, S→5, B→8, dll.
    """
    from collections import Counter as _Ctr

    # Tabel translasi karakter OCR → digit
    # FROM: O o l L I i s S b B z Z G g A a D d Y y P p F f C c U u V v ?
    # TO  : 0 0 1 1 1 1 5 5 8 8 2 2 6 6 0 0 0 0 4 4 9 9 0 0 0 0 0 0 0 0 7
    _KK_FROM = "OolLIisSbBzZGgAaDdYyPpFfCcUuVv?"
    _KK_TO   = "0011115588226600004499000000007"
    _OCR_KK  = str.maketrans(_KK_FROM, _KK_TO)

    def _to_16digits(raw: str) -> str:
        """
        Bersihkan raw OCR string → 16 digit nomor KK.

        Menangani kasus-kasus noise khas font besar KK:
          1. Spasi antar digit: "7 1 1 9 0 9 2 0 2 4" → "7119092024"
          2. Karakter noise di DEPAN nomor: "?2140120212753" — karakter '?'
             adalah artefak dari logo/ornamen KK yang OCR-baca sebagai simbol.
             Strategi: coba strip 1-2 karakter non-digit di awal sebelum translasi,
             lalu periksa apakah hasilnya 16 digit valid (kode provinsi 11-99).
          3. Karakter OCR salah baca: O→0, l→1, I→1, S→5, B→8, dll.
        """
        def _clean_and_count(s: str) -> str:
            """Bersihkan satu string raw → digit, kembalikan string digit atau ''."""
            # Tahap 1: compact spasi antar karakter (font besar)
            compacted = re.sub(r"(?<=[0-9A-Za-z?])\s(?=[0-9A-Za-z?])", "", s)
            # Tahap 2: hapus non-alfanumerik kecuali ?
            cleaned = re.sub(r"[^0-9A-Za-z?]", "", compacted)
            # Tahap 3: translasi OCR noise → digit
            digits = re.sub(r"[^0-9]", "", cleaned.translate(_OCR_KK))
            return digits

        digits = _clean_and_count(raw)

        # Kasus tepat 16 digit → langsung terima
        if len(digits) == 16 and re.match(r"[1-9][0-9]", digits[:2]):
            return digits

        # Kasus >16 digit: mungkin ada prefix noise yang ikut terhitung.
        # Coba buang 1 atau 2 digit pertama hasil translasi (noise prefix seperti
        # "?" → "7") dan cek apakah sisanya 16 digit dengan kode provinsi valid.
        if len(digits) > 16:
            for skip in (1, 2):
                candidate = digits[skip:]
                if len(candidate) == 16 and re.match(r"[1-9][0-9]", candidate[:2]):
                    return candidate
            # Jika tidak ada yang lolos, ambil 16 pertama sebagai last resort
            if re.match(r"[1-9][0-9]", digits[:2]):
                return digits[:16]

        # Kasus 15 digit: satu digit terpotong → kandidat lemah
        if len(digits) == 15 and re.match(r"[1-9][0-9]", digits[:2]):
            return digits + "0"

        return ""

    candidates = []

    # ── Pola A-1: label eksplisit "NOMOR KK" atau "No. KK" ──────────────────
    # Paling andal — OCR PSM6/11 biasanya membaca label ini dengan benar.
    m1 = re.search(
        r"(?:NOMOR\s+KK|No\.?\s*KK|N[o0]\.\s*KK)\s*[:\-]?\s*"
        r"([0-9A-Za-z?][0-9A-Za-z?\s]{10,28})(?:\n|$|[^0-9A-Za-z?\s])",
        text, re.IGNORECASE,
    )
    if m1:
        result = _to_16digits(m1.group(1).strip())
        if result:
            candidates.append(result)

    # ── Pola A-2: baris yang dimulai "No." atau "No.-" diikuti digit/noise ──
    # Dari debug: nomor KK dibaca sebagai baris tersendiri:
    #   "No."          (baris 1 — label)
    #   "?2140120212753" (baris 2 — isi nomor, dengan noise "?" di depan)
    # Tangani dua sub-pola:
    #   A-2a: label dan nomor pada SATU baris: "No. ?2140120212753"
    #   A-2b: label pada satu baris, nomor pada baris BERIKUTNYA (PSM11 split)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # A-2a: satu baris — "No[.-] <noise><digit...>"
        # Izinkan 0-2 karakter noise (seperti ?, #, *, simbol) sebelum digit pertama
        m2a = re.match(
            r"N[o0u][a\.\-]?\s*[:\-]?\s*([?#*@!~\^$]{0,2}[0-9][0-9A-Za-z?\s]{11,26})$",
            line_stripped, re.IGNORECASE,
        )
        if m2a:
            result = _to_16digits(m2a.group(1).strip())
            if result:
                candidates.append(result)
            continue  # jangan cek A-2b untuk baris ini

        # A-2b: baris hanya berisi "No." atau "No.-" → lihat baris berikutnya
        # (PSM11 memisahkan label dan nilai ke baris berbeda karena font besar)
        if re.match(r"^N[o0u][a\.\-]?\s*[:\-]?\s*$", line_stripped, re.IGNORECASE):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Baris berikutnya boleh dimulai noise 1-2 karakter lalu digit
                m2b = re.match(
                    r"([?#*@!~\^$]{0,2}[0-9][0-9A-Za-z?\s]{11,26})$",
                    next_line,
                )
                if m2b:
                    result = _to_16digits(m2b.group(1).strip())
                    if result:
                        candidates.append(result)

    # ── Pola B: fallback — baris murni angka tanpa label ────────────────────
    # Aktif HANYA jika Pola A tidak menghasilkan kandidat.
    # Filter ketat: skip baris yang mengandung kata field KTP/NIK/anggota keluarga.
    _KTP_HINTS = re.compile(
        r"\b(?:NIK|Nomor\s+Induk|Tempat|Lahir|Berlaku|Kecamatan|Kelurahan"
        r"|Provinsi|Kabupaten|Golongan|Status|Pekerjaan|Kewarganegaraan)\b",
        re.IGNORECASE,
    )
    if not candidates:
        for line in lines:
            if _KTP_HINTS.search(line):
                continue
            for m3 in re.finditer(
                r"(?<![A-Za-z0-9])[?#*]{0,2}[1-9][0-9A-Za-z?]{13,17}(?![A-Za-z0-9])",
                line,
            ):
                result = _to_16digits(m3.group(0))
                if result:
                    candidates.append(result)

    if not candidates:
        return ""

    # Voting: kandidat terbanyak menang.
    winner, _ = _Ctr(candidates).most_common(1)[0]
    return winner


def _extract_nama_kepala_kk(text: str) -> str:
    """
    Ekstrak nama kepala keluarga dari KK .

    Strategi berlapis:
      1. Label "Nama Kepala Keluarga" (PSM6/11 — ada label eksplisit)
      2. Baris setelah nomor KK (PSM3 — tanpa label, nama langsung di bawah nomor)
      3. Baris ALL-CAPS 2-4 kata di bagian atas dokumen
    """
    # Pola label "Nama Kepala Keluarga" — toleran terhadap noise OCR umum
    # "Kopata Kelunga", "Kodata Kotinga", "Kopat Kairon", "Nara Kapala Kolirga", dst.
    KK_LABEL = (
        r"(?:Nama\s+)?"
        r"(?:Kepala|Kopala|Kopata|Kodata|Kopaia|Kofata|Kapala|Kapata|Kopat|"
        r"Nara\s+Kapala|SEK\s+Namis|Kopsia|Kofuarya|Kopat)\s+"
        r"(?:Keluarga|Kotarga|Kofuarga|Kovarya|Kovarga|Kelinrgai|Kekiarya|"
        r"Kotinga|Ketiarga|Kelirga|Kolatoa|Koturga|Koluarga|Kaliarga|"
        r"Kotlarga|Kairon|Karol|Kainon|Karya|Kairga|Kolarga)"
    )
    SKIP = {
        "desa","desm","kelurahan","kecamatan","kabupaten","kota",
        "desafkelurahan","desmkolurehan","no","nomor",
        "desnfaturahan","desn","padanaan","padanman",
    }
    # Kata-kata yang BUKAN nama orang — termasuk varian noise label KK
    GEO = {
        "mekarwangi","cikarang","barat","timur","bekasi","jakarta","bandung",
        "kelurahan","kecamatan","kabupaten","provinsi","jawa","keluarga",
        "kartu","desa","no","nomor","rt","rw","surabaya","medan",
        "tangerang","depok","bogor","karawang",
        # Noise OCR dari kata "Kepala"
        "kopata","kodata","kopaia","kofata","kapala","kopala","kapata",
        "kopsia","koturga","kotinga","kelunga","kotarga","kofuarga",
        "kovarya","kovarga","kelinrgai","kekiarya","koluarga","kaliarga",
        "kotlarga","kolatoa","ketiarga","kelirga","kopat","kairon",
        # Noise PSM11 khas (kata acak yang bukan nama)
        "direnfaturaban","desafkelurahan","desmkolurehan",
        # Penggalan kata geografi yang muncul sebagai OCR noise
        "kecam","kelur","kabup","provin","miahan","sioaraua","sidarua",
        "sidaraia","buana","kairon","karol","kainon","padanaan","padanman",
        "sumedang","paseh","bekasi","bandung","jakarta","surabaya",
        # v5.7.6-KK: Noise OCR dari label "Desa/Kelurahan" yang sering tercampur
        # ke dalam baris nama kepala keluarga (GesalKeuran, Gesalkarang, dst)
        "gesalkarang","gesalkeuran","gesalkeluran","desalkeluran",
        "gesalk","desalk","keuran","keluran","keuran",
        # Nama geografi umum lainnya
        "maju","sejahtera","bangun","makmur","mandiri","sentosa",
    }

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── Strategi 1: label resmi ───────────────────────────────────────────
    for i, line in enumerate(lines):
        if not re.search(KK_LABEL, line, re.IGNORECASE):
            continue
        parts = re.split(KK_LABEL + r"\s*[:\-—\?]?\s*",
                         line, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            cand = parts[1]
            # Potong di kata geografi / noise yang mengikuti nama
            cand = re.split(
                r"\s*[:/]\s*(?:Desa|Desm|Kel|Kec|Kabupaten|Provinsi|Desnfaturahan|Desn)",
                cand, flags=re.IGNORECASE
            )[0]
            cand = re.split(r"\s*[—–]\s*", cand)[0]
            # Hapus karakter non-huruf lalu ambil kata ≥2 huruf yang bukan noise
            words = [w for w in re.sub(r"[^A-Za-z\s]", " ", cand).split()
                     if len(w) >= 2 and w.lower() not in SKIP and w.lower() not in GEO]
            if words and len(words) <= 4:
                slug = "_".join(w.lower() for w in words)
                if _is_plausible_name(slug):
                    return " ".join(words[:4]).upper()
        # Coba baris berikutnya jika tidak ada konten di baris yang sama
        if i + 1 < len(lines):
            nxt   = lines[i + 1]
            words = [w for w in re.sub(r"[^A-Za-z\s]", " ", nxt).split()
                     if len(w) >= 3 and w.lower() not in GEO]
            # Nama orang butuh minimal 1 kata dgn ≥5 huruf atau 2+ kata
            if (words and len(words) <= 4
                    and not any(w.lower() in SKIP for w in words)
                    and not re.search(r"\d", nxt)
                    and (len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 5))):
                slug = "_".join(w.lower() for w in words)
                if _is_plausible_name(slug):
                    return " ".join(words).upper()

    # ── Strategi 2: nama di baris setelah nomor KK (PSM3 format) ─────
    for i, line in enumerate(lines):
        is_nomor = (
            re.search(r"N[oua][.\-\s]*[:\-]?\s*[0-9?(]", line, re.IGNORECASE) or
            re.search(r"[0-9]{4}[0-9A-Za-z?)(]{10,}", line)
        ) and re.search(r"[0-9]{4,}", line)
        if not is_nomor:
            continue
        for j in range(i + 1, min(i + 10, len(lines))):
            cand = lines[j]
            cand = re.sub(r"^[^A-Za-z]+", "", cand).strip()
            if not cand:
                continue
            # Potong di kata geografi yang mengikuti nama (pola PSM3)
            cand = re.split(
                r"\s+(?:Desa|Desai|Desaf|Desm|Dexa|Desnfaturahan|Kelurahan|Desn)",
                cand, flags=re.IGNORECASE
            )[0]
            cand = cand.split(":")[0].strip()
            # Ambil hanya kata-kata di depan sebelum noise dash
            cand = re.split(r"\s*[—–]\s*", cand)[0].strip()
            words = re.findall(r"[A-Za-z]{2,}", cand)
            if not (1 <= len(words) <= 5):
                continue
            if any(w.lower() in GEO for w in words):
                continue
            if re.search(r"\d{3,}", cand):
                continue
            # Validasi dengan _is_plausible_name — tolak kata noise pendek
            slug = "_".join(w.lower() for w in words)
            if not _is_plausible_name(slug):
                continue
            return " ".join(w.upper() for w in words[:4])

    # ── Strategi 3: baris ALL-CAPS 2-4 kata di 12 baris pertama ──────
    # Blacklist gabungan: geo + varian noise label KK
    GEO_ALL = GEO | {
        "KELUARGA","KARTU","ARTU","PROVINSI","JAWA","BARAT",
        "TIMUR","SELATAN","UTARA","TENGAH","PEMERINTAH","REPUBLIK",
        "INDONESIA","DIRENFATURABAN","DESAFKELURAHAN","DESMKOLUREHAN",
        "KOPATA","KODATA","KOPAIA","KOFATA","KAPALA","KOPALA","KAPATA",
        "KOPSIA","KOTURGA","KOTINGA","KELUNGA","KOTARGA","KOLATOA",
    }
    for line in lines[:12]:
        cand = re.sub(r"^[^A-Za-z]+", "", line).strip()
        cand = re.split(
            r"\s+(?:Desa|Desai|/Kelurahan|Kecamatan)",
            cand, flags=re.IGNORECASE
        )[0]
        cand = cand.split(":")[0].strip()
        words = re.findall(r"[A-Za-z]{3,}", cand)
        if not (2 <= len(words) <= 4):
            continue
        if not all(w[0].isupper() for w in words):
            continue
        if any(w.upper() in GEO_ALL for w in words):
            continue
        if re.search(r"\d", cand):
            continue
        return " ".join(w.upper() for w in words)

    # ── Strategi 4: nama di baris setelah noise label (pola PSM11) ───────────
    # Menangani: "Kopata Kelunga\nKusnadi" dan "Direnfaturaban\n...\nKopata Kelunga\nKusnadi"
    LABEL_NOISE = re.compile(
        r"(?:Kopata|Kodata|Kopaia|Kofata|Kapala|Kopala|Kapata|Kopsia|"
        r"Koturga|Kotinga|Kelunga|Kotarga|Kofuarga|Kovarya|Kovarga|"
        r"Kelinrgai|Kekiarya|Koluarga|Kaliarga|Kotlarga|Kolatoa|Ketiarga|"
        r"Kelirga|Nara\s+Kapala|SEK\s+Namis|Direnfaturaban)",
        re.IGNORECASE
    )
    # Semua kata yang BUKAN bagian dari nama orang
    GEO_EXTENDED = GEO_ALL | {
        "DUSUN","DSN","DSUN","JALAN","GANG","BLOK","PERUM","KOMP","KOMPLEK",
        "GRIYA","PERUMAHAN","KAVLING","KAV","KAMPUNG","KP","SIOARAUA",
        # Kata noise label KK yang mungkin lolos sebagai kata tersendiri
        "KOPATA","KODATA","KOPAIA","KOFATA","KAPALA","KOPALA","KAPATA",
        "KOPSIA","KOTURGA","KOTINGA","KELUNGA","KOTARGA","KOFUARGA",
        "KOVARYA","KOVARGA","KELINRGAI","KEKIARYA","KOLUARGA","KALIARGA",
        "KOTLARGA","KOLATOA","KETIARGA","KELIRGA",
        "DIRENFATURABAN","DESAFKELURAHAN","DESMKOLUREHAN","MAAN",
    }
    found_noise_at = None
    for i, line in enumerate(lines):
        if LABEL_NOISE.search(line):
            found_noise_at = i
            break

    if found_noise_at is not None:
        # Cari dari baris noise sampai maks 8 baris ke bawah
        # Ambil baris pertama yang: semua kata bukan GEO_EXTENDED, ada ≥1 kata, lulus plausible
        for j in range(found_noise_at + 1, min(found_noise_at + 9, len(lines))):
            nxt   = lines[j]
            words = re.findall(r"[A-Za-z]{2,}", nxt)
            if not words or len(words) > 4:
                continue
            if re.search(r"\d{2,}", nxt):
                continue
            if any(w.upper() in GEO_EXTENDED for w in words):
                continue
            slug = "_".join(w.lower() for w in words)
            if _is_plausible_name(slug):
                return " ".join(w.upper() for w in words[:4])

    # ── Strategi 5: baris tabel KK panjang (format OCR multi-kolom) ─────────
    # KK modern: OCR sering mengembalikan seluruh baris tabel anggota sebagai
    # satu baris panjang, misal:
    # "1 DADANG SUHENDA 3216082501030001 L BEKASI 26-01-1975 ISLAM KAWIN KEPALA KELUARGA"
    # Strategi sebelumnya gagal karena ada digit (NIK) dalam baris.
    # Strategi ini: ekstrak nama dari baris tabel yang diawali "1" atau "1."
    # dan mengandung "KEPALA KELUARGA" sebagai konfirmasi hubungan keluarga.
    for line in lines:
        # Baris harus diawali nomor urut "1" (kepala keluarga selalu anggota pertama)
        if not re.match(r"^\s*1[\s\.\|]+", line):
            continue
        # Harus ada penanda hubungan KEPALA KELUARGA di baris yang sama
        if not re.search(r"KEPALA\s+KELUARGA", line, re.IGNORECASE):
            continue
        # Potong baris: ambil bagian setelah nomor urut sampai sebelum NIK/digit panjang
        # Contoh: "1 DADANG SUHENDA 321608..." → cari kata-kata sebelum sequence digit ≥6
        after_num = re.sub(r"^\s*1[\s\.\|]+", "", line).strip()
        # Ambil semua kata di depan sebelum sequence digit panjang (NIK)
        parts = re.split(r"\s+\d{5,}", after_num)
        if not parts:
            continue
        candidate = parts[0].strip()
        words = [w for w in re.findall(r"[A-Za-z]{2,}", candidate)
                 if w.lower() not in GEO]
        if not (1 <= len(words) <= 5):
            continue
        slug = "_".join(w.lower() for w in words)
        if _is_plausible_name(slug):
            return " ".join(w.upper() for w in words[:4])

    return ""


def _extract_alamat_kk(text: str, all_src: list = None) -> str:
    """
    Ekstrak alamat dari KK dari baris setelah label 'Alamat'
    atau baris yang mengandung kata kunci alamat (PERUM, JL., KP., BLOK, GANG).
    """
    candidates = []
    for src in (all_src or [text]):
        lines = [l.strip() for l in src.split("\n") if l.strip()]
        for i, line in enumerate(lines):
            if re.search(r"\bAlamat\b", line, re.IGNORECASE):
                # Ambil konten setelah 'Alamat :' di baris yang sama
                after = re.split(r"Alamat\s*:?\s*", line, flags=re.IGNORECASE, maxsplit=1)
                if len(after) > 1 and len(after[1].strip()) > 3:
                    raw = after[1]
                else:
                    raw = lines[i + 1] if i + 1 < len(lines) else ""
                if raw:
                    cleaned = _clean_alamat_kk(raw)
                    if cleaned:
                        candidates.append(cleaned)
                break
            # Fallback: baris dengan kata kunci alamat (termasuk DUSUN/DSUN)
            if re.search(r"(?:PERUM|JL|JALAN|KP|BLOK|GANG|GG|DUSUN|DSN|DSUN)", line, re.IGNORECASE):
                cleaned = _clean_alamat_kk(line)
                if cleaned:
                    candidates.append(cleaned)
                    break

    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


def _clean_alamat_kk(raw: str) -> str:
    """Bersihkan noise dari string alamat KK, pertahankan PERUM/JL dst."""
    m = re.search(r"(?:PERUM|JL|JALAN|KP|GANG|GG|DUSUN|DSN|DSUN).+", raw, re.IGNORECASE)
    if m:
        alamat = m.group(0)
        # Potong di keyword geografi
        alamat = re.split(
            r"\s+(?:Kecamatan|Kelurahan|Kabupaten|Kota)",
            alamat, flags=re.IGNORECASE,
        )[0]
        return re.sub(r"[^A-Za-z0-9\s\./]", " ", alamat).strip()
    # Jika tidak ada kata kunci alamat, bersihkan saja teks mentah
    cleaned = re.sub(r"[^A-Za-z0-9\s\./]", " ", raw).strip()
    return cleaned if len(cleaned) > 4 else ""


def _extract_rtrw(text: str) -> str:
    """
    Ekstrak RT/RW dari KK (v5.4).

    Pola yang ditangani:
      1. "RT.045/RW.011"        — eksplisit
      2. "RTRW 0451011"         — 7-digit, smart split
      3. "RTRW\\nOasiots"       — RTRW + newline
      4. "005/001 Kabupaten"    — format PSM3 (tanpa label RTRW)
      5. "00SI0U1 Kabupaten"    — format PSM3 tanpa "/" + noise OCR
    """
    _OCR2 = str.maketrans("OolLIisSaA", "0011115500")

    def _fix(raw):
        return re.sub(r"[^0-9]", "", raw.translate(_OCR2))

    def _fix_rtrw(raw):
        """OCR mapping khusus RT/RW: t→1 (bukan 0), U→0, G→0."""
        MAP = {
            "O":"0","o":"0","l":"1","L":"1","I":"1","i":"1",
            "S":"5","s":"5","U":"0","u":"0","G":"0","g":"0",
            "t":"1","T":"1",
        }
        return "".join(MAP.get(c, c if c.isdigit() else "") for c in raw)

    def _parse(d):
        if len(d) < 4:
            return ""
        if len(d) == 6:
            return f"{d[:3]}/{d[3:]}"
        if len(d) == 7:
            for rt, rw in [(d[:3], d[4:]), (d[1:4], d[4:]), (d[:3], d[3:6])]:
                try:
                    if 1 <= int(rt) <= 200 and 1 <= int(rw) <= 50:
                        return f"{rt}/{rw}"
                except ValueError:
                    continue
            return f"{d[:3]}/{d[3:6]}"
        return f"{d[:3]}/{d[3:6]}" if len(d) >= 6 else ""

    # Pola 1: RT/RW eksplisit dengan pemisah /, spasi, atau noise OCR
    m = re.search(r"RT[/I\s.]*RW[.\s:]*([\d]{2,3})[/\s\-]([\d]{2,3})",
                  text, re.IGNORECASE)
    if m:
        return f"{m.group(1).zfill(3)}/{m.group(2).zfill(3)}"

    # Pola 1b: RT/RW eksplisit dengan pemisah . (dot)
    m1b = re.search(r"RT\.?([\d]{2,3})/RW\.?([\d]{2,3})", text, re.IGNORECASE)
    if m1b:
        return f"{m1b.group(1).zfill(3)}/{m1b.group(2).zfill(3)}"

    # Pola 2: RTRW/BURW/RTIRW satu baris
    # v5.7.6-KK: tambah 'RTIRW' (noise OCR dari 'RT/RW' dengan huruf I sisipan)
    m2 = re.search(r"(?:RTRW|RTIRW|BURW|RT[I1]?RW)\s*[:\-]?\s*([0-9A-Za-z]{4,10})\b",
                   text, re.IGNORECASE)
    if m2:
        r = _parse(_fix(m2.group(1)))
        if r:
            return r

    # Pola 3: RTRW/BURW/RTIRW + newline
    m3 = re.search(r"(?:RTRW|RTIRW|BURW|RT[I1]?RW)\s*\n\s*([0-9A-Za-z]{4,10})",
                   text, re.IGNORECASE)
    if m3:
        r = _parse(_fix(m3.group(1)))
        if r:
            return r

    # Pola 4: "NNN/NNN Kabupaten" — format KK PSM3 dengan separator "/"
    for line in text.split("\n"):
        m4 = re.search(r"([0-9A-Za-z]{2,4})\s*/\s*([0-9A-Za-z]{2,4})", line)
        if not m4:
            continue
        rt_d = _fix_rtrw(m4.group(1))
        rw_d = _fix_rtrw(m4.group(2))
        if len(rt_d) < 2 or len(rw_d) < 2:
            continue
        try:
            if (1 <= int(rt_d) <= 200 and 1 <= int(rw_d) <= 50
                    and re.search(r"Kabupaten|Kota|BEKASI|JAKARTA", line, re.IGNORECASE)):
                return f"{rt_d.zfill(3)}/{rw_d.zfill(3)}"
        except ValueError:
            pass

    # Pola 5: "NNNNNNN Kabupaten/Kecamatan" — format PSM3 tanpa "/" (6-8 digit menyatu)
    for line in text.split("\n"):
        m5 = re.search(r"([0-9A-Za-z]{6,8})\s+(?:Kabupaten|Kecamatan)", line, re.IGNORECASE)
        if not m5:
            continue
        d = _fix_rtrw(m5.group(1))
        r = _parse(d)
        if r:
            return r

    return ""

# ─── Fungsi baru: Ekstraksi field header KK (v5.5) ────────────────────────────

def _extract_kk_header_field(text: str, field: str) -> str:
    """
    Ekstrak satu field dari blok header KK dengan toleransi noise OCR (v5.5).

    Layout header KK Indonesia (di bawah nomor KK):
      Desa/Kelurahan  : MEKARWANGI
      Kecamatan       : CIKARANG BARAT
      Kabupaten/Kota  : BEKASI
      Provinsi        : JAWA BARAT
      RT/RW           : 005/001
      Alamat          : KP RAWA BANTENG

    Format PSM3 (tanpa label eksplisit) ditangani dengan pola inline:
      "DADANG SUHENDA: Desa/Kelurahan: MEKARWANGI Kecamatan CIKARANG BARAT"

    Args:
        text  (str): Teks OCR yang sudah dibersihkan.
        field (str): Nama field — 'desa_kelurahan', 'kecamatan',
                     'kabupaten_kota', atau 'provinsi'.

    Returns:
        str: Nilai field UPPERCASE, atau "" jika tidak ditemukan.
    """
    # Perbaikan v5.7.6-KK: Ubah [A-Z] → [A-Za-z] di semua pola value agar
    # field terbaca meskipun OCR menghasilkan nilai lowercase (misal "mekarwangi").
    # Tidak mempengaruhi KTP/SIM karena fungsi ini hanya dipanggil untuk doc_type=="kk".
    #
    # PENTING: Pola desa_kelurahan TIDAK boleh menggunakan '(?:Desa|Kel)' karena
    # 'Kel' juga muncul dalam 'Keluarga' (baris kepala keluarga) dan menghasilkan
    # false positive seperti 'PASPOR' (dari 'warganegara No. paspo!').
    # Gunakan pola yang lebih spesifik: 'Desa\s*/\s*Kelurahan' atau 'Desaf?kelurahan'.
    PATTERNS = {
        "desa_kelurahan": [
            # Pola label eksplisit — tidak menggunakan (?:Kel) sendiri (false positive!)
            r"Desa\s*/\s*Kelurahan\s*[:\-=.\s]\s*([A-Za-z][A-Za-z\s]{2,30}?)(?=\s+(?:Kec|Kab|Prov|No\b)|[\n$])",
            r"Desa\s*/\s*Ke[lt]urahan\s*[:\-=1.]\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\s+(?:Kec|Kab|Prov)|\n|$)",
            r"Dexa\s*/\s*Kelurahan\s*[:\-—.]?\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\s+(?:Kec|Kab)|\n|$)",
            r"Desaf?kelurahan\s*[:\-—#.]?\s*([A-Za-z][A-Za-z]{2,20})",
            # Noise OCR dari 'Desa/Kelurahan': GesalKeuran, Gesalkarang, GesalKeluran
            # Diikuti tanda pisah/spasi lalu nilai desa
            r"(?:Gesal[A-Za-z]{3,12}|Desalk[A-Za-z]{2,10})\s*[.\s,]*\s+([A-Z][A-Za-z\s]{2,25}?)(?=\n|$|\s+(?:Kec|Kab))",
            # Fallback: 'DESA [NAMA]' sebagai standalone keyword (uppercase)
            # HANYA match jika DESA diikuti kata kapital (bukan noise lowercase)
            r"(?<![A-Za-z])DESA\s+([A-Z][A-Za-z]{2,20}(?:\s+[A-Z][A-Za-z]{2,20})?)(?=\n|$|\s+(?:Kec|Kab|Prov))",
        ],
        "kecamatan": [
            r"Kecamatan\s*[:\-=.]?\s*([A-Za-z][A-Za-z\s]{2,30}?)(?=\s+(?:Kab|Kota|Prov)|[\n$])",
            r"Kecamatan\s*[:\-=.]?\s*1?\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\s+(?:Kab|Kota)|\n|$)",
            r"Kecarnatan\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\s+(?:Kab|Kota)|\n|$)",
            r"(?<![A-Za-z])Kec\b\.?\s*[:\-.]?\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\s+(?:Kab|Kota)|\n|$)",
            # v5.7.6-KK: shorthand 'KEC.' yang sering muncul di KK tercetak
            r"(?<![A-Za-z])KEC\.?\s+([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$|\s+(?:Kab|Kota|Prov))",
        ],
        "kabupaten_kota": [
            r"Kabupaten\s*/?\s*Kota\s*[:\-=.]?\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$|\s+Prov)",
            r"Kabupaten\s*/?\s*Kota\s*:\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$|\s+Prov)",
            r"Kabupaton\s*/?\s*Kota\s*([A-Za-z][A-Za-z\s]{2,20}?)(?=\n|$|\s+Prov)",
            r"Kab\.?\s*/?\s*Kota\s*[:\-.]?\s*([A-Za-z][A-Za-z\s]{2,20}?)(?=\n|$|\s+Prov)",
            r"Kabupaten\s*[:\-.]?\s*([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$|\s+Prov)",
            # v5.7.6-KK: shorthand 'KAB.' yang sering muncul standalone di teks KK
            r"(?<![A-Za-z])KAB\.?\s+([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$|\s+(?:Prov|Kode))",
        ],
        "provinsi": [
            # v5.7.6-KK: Nilai provinsi bisa diawali digit noise (mis: '3ANA BARAT' = 'JAWA BARAT')
            # Pola \d? di depan value menangkap digit opsional agar tidak memblokir ekstraksi.
            r"[Pp]rovinsi\s*[:\-=.1]?\s*\d?([A-Za-z][A-Za-z\s]{2,30}?)(?=\s*(?:No\.?\s*\d|Jenis|Nama\s+Lengkap|$)|\n)",
            r"Provins[il]\s*[:\-=.]?\s*\d?([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$)",
            r"Prov\.\s*[:\-.]?\s*\d?([A-Za-z][A-Za-z\s]{2,25}?)(?=\n|$)",
            # Fallback: label pendek 'Ansi', 'Anis' (noise OCR dari 'Provinsi')
            r"(?:Ansi|Anis|Provins)\s*[.:\-]?\s*([A-Za-z][A-Za-z\s]{3,30}?)(?=\n|$)",
        ],
    }

    # Kata geografi yang tidak boleh menjadi nilai field
    GEO_STOP = {
        "KECAMATAN", "KABUPATEN", "KOTA", "PROVINSI", "DESA",
        "KELURAHAN", "RT", "RW", "RTRW", "ALAMAT", "NO", "NOMOR",
        "JENIS", "GOLONGAN", "NAMA", "PEKERJAAN",
    }

    pats = PATTERNS.get(field, [])
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            # Bersihkan noise di akhir (angka, titik dua, tanda baca)
            val = re.sub(r"\s*[:\-—#\d\.]\s*$", "", val).strip()
            # Hapus kata geografi di akhir
            words = [w for w in val.split() if w.upper() not in GEO_STOP]
            val   = " ".join(words).strip()
            if len(val) >= 3:
                return val.upper()
    return ""


def _extract_kk_fields_voting(all_src: list) -> dict:
    """
    Voting semua field header KK dari seluruh varian OCR (v5.5).

    Mengekstrak desa_kelurahan, kecamatan, kabupaten_kota, dan provinsi
    dari setiap varian OCR lalu memilih nilai terbanyak.

    Args:
        all_src (list[str]): Semua varian teks OCR.

    Returns:
        dict: {field: nilai_terpilih, ...}
    """

    kk_fields = ["desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi"]
    result    = {}

    for field in kk_fields:
        candidates = [_extract_kk_header_field(t, field) for t in all_src]
        candidates = [v for v in candidates if v]
        if candidates:
            result[field] = Counter(candidates).most_common(1)[0][0]
        else:
            result[field] = ""

    return result


# ─── Subfungsi Ekstraksi Nama ─────────────────────────────────────────────────

def _extract_name(
    text: str,
    doc_type: str,
    all_texts: list = None,
) -> str:
    """
    Mengekstrak nama pemilik dokumen menggunakan 4 strategi berlapis.

    Strategi dijalankan pada SEMUA 12 varian OCR (all_texts), menghasilkan
    pool kandidat. Kandidat terbaik dipilih berdasarkan jumlah kata terbanyak
    (nama lengkap lebih diprioritaskan daripada nama pendek).

    Strategi (diurutkan dari paling spesifik ke paling umum):
      1. Label "Nama :"   — split berdasarkan karakter ':' pada baris "Nama"
      2. Header KK        — split dari baris "Nama Kepala Keluarga :"
      3. Setelah NIK      — nama selalu muncul 1-3 baris setelah NIK (16 digit)
                            pada dokumen KTP/KK nyata (ditemukan dari debug data)
      4. Setelah POLRI    — nama muncul setelah baris "POLRI" pada SIM

    Args:
        text      (str)       : Teks OCR terbaik.
        doc_type  (str)       : Jenis dokumen untuk memilih strategi.
        all_texts (list[str]) : Semua varian OCR untuk voting.

    Returns:
        str: Nama dalam format slug lowercase_underscore, atau "" jika gagal.
    """
    candidates = []
    sources    = [text] + (all_texts or [])

    for src in sources:
        n = _name_from_label(src)
        if n:
            candidates.append(n)

        if doc_type == "kk":
            n = _name_from_kk_header(src)
            if n:
                candidates.append(n)

        if doc_type in ("ktp", "kk"):
            n = _name_after_nik(src)
            if n:
                candidates.append(n)
            else:
                # Strategi 3b: baris ALL-CAPS ketika NIK tidak terbaca
                n_caps = _name_all_caps_scan(src)
                if n_caps:
                    candidates.append(n_caps)

        if doc_type == "sim":
            n = _name_after_polri(src)
            if n:
                candidates.append(n)

    candidates = [c for c in candidates if c and _is_plausible_name(c)]
    if not candidates:
        return ""

    # Pilih kandidat menggunakan majority vote (paling sering muncul),
    # dengan tiebreaker: kandidat lebih panjang diprioritaskan.
    # Perbaikan dari max(kata terbanyak) yang rentan memilih string alamat.
    freq = Counter(candidates)
    max_freq = freq.most_common(1)[0][1]
    top = [c for c, f in freq.items() if f == max_freq]
    return max(top, key=lambda n: (len(n.split("_")), len(n)))


def _name_from_label(text: str) -> str:
    """
    Strategi 1: Cari baris mengandung "Nama" lalu ambil konten setelah ':'.

    Abaikan baris "Nama Kepala Keluarga" — ditangani oleh _name_from_kk_header.
    Toleran terhadap noise: "NA MA", "NAM4", "NAMA" semua cocok dengan
    pattern r"NA\\s*MA?".
    """
    for line in text.split("\n"):
        upper = line.upper()
        if re.search(r"\bNA\s*MA?\b", upper):
            if re.search(r"KEPALA\s+KELUARGA", upper):
                continue  # lewati baris khusus KK
            parts = re.split(r"[::]", line, maxsplit=1)
            if len(parts) == 2:
                name = _clean_name(parts[1])
                if name:
                    return name
            # Fallback: ambil kata setelah "Nama" tanpa ':'
            m = re.search(r"NA\s*MA?\s+([A-Za-z ]{3,})", line, re.IGNORECASE)
            if m:
                name = _clean_name(m.group(1))
                if name:
                    return name
    return ""


def _name_from_kk_header(text: str) -> str:
    """
    Strategi 2: Cari baris "Nama Kepala Keluarga :" — label eksklusif KK.

    Label ini hanya ada di KK sehingga sangat andal untuk ekstraksi nama
    kepala keluarga.
    """
    for line in text.split("\n"):
        if re.search(r"KEPALA\s+KELUARGA", line.upper()):
            parts = re.split(r"[::]", line, maxsplit=1)
            if len(parts) == 2:
                name = _clean_name(parts[1])
                if name:
                    return name
    return ""


def _name_after_nik(text: str) -> str:
    """
    Strategi 3 : Nama muncul 1-5 baris SETELAH baris NIK.

    Perbaikan dari v5.1:
      - Window diperlebar ke +5 baris (KTP kadang ada baris noise setelah NIK)
      - Skip baris yang rasio noise-nya tinggi (>40% non-huruf)
      - Skip baris yang mayoritas berisi stopwords

    Pola NIK yang ditoleransi: 14-18 alfanumerik dengan noise OCR
    (O→0, l→1, S→5, b→6, D→0, dll) atau label "NIK".
    """
    NOISE_STOPWORDS = {
        "kabupaten", "kota", "kecamatan", "kelurahan", "desa",
        "pelajar", "mahasiswa", "status", "perkawinan", "pekerjaan",
        "kewarganegaraan", "wni", "alamat", "darah", "berlaku",
        "nama", "tempat", "lahir", "provinsi", "republik", "indonesia",
    }
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, line in enumerate(lines):
        # Deteksi baris NIK: sequence alfanumerik 14-18 atau label NIK
        cleaned_line = re.sub(r"[\s\-]", "", line)
        is_nik_line  = (
            (len(cleaned_line) >= 14 and
             len(re.findall(r"[0-9]", cleaned_line.translate(
                 str.maketrans("OolLIisSbBzZGgAaDd", "001111558822660044")
             ))) >= 13) or
            re.search(r"\bN\s*[I1]\s*K\b", line.upper())
        )
        if is_nik_line:
            for j in range(i + 1, min(i + 6, len(lines))):
                candidate = lines[j]
                # Skip baris noise: rasio non-huruf tinggi
                alpha = re.findall(r"[A-Za-z]", candidate)
                if not alpha:
                    continue
                noise_ratio = len(re.findall(r"[^A-Za-z\s]", candidate)) / max(1, len(candidate))
                if noise_ratio > MAX_NOISE_RATIO:
                    continue
                # Skip baris stopwords
                words_lower = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", candidate)]
                if any(w in NOISE_STOPWORDS for w in words_lower):
                    continue
                name = _clean_name(candidate)
                if name and _is_plausible_name(name):
                    return name
    return ""



def _name_all_caps_scan(text: str) -> str:
    """
    Strategi 3b: Cari nama dari baris ALL-CAPS 2-4 kata di 15 baris pertama.

    Digunakan sebagai cadangan ketika NIK tidak terbaca (misal adap PSM11
    yang tidak menampilkan NIK secara eksplisit, tapi DIPCA ANUGRAH terbaca).

    Kriteria baris:
      - Semua huruf KAPITAL
      - 2-4 kata, masing-masing >= 3 huruf
      - Tidak ada angka
      - Tidak ada stopword nama
    """
    CAPS_STOP = {
        "kabupaten", "kota", "kecamatan", "kelurahan", "desa", "provinsi",
        "republik", "indonesia", "pelajar", "mahasiswa", "pekerjaan",
        "perkawinan", "kewarganegaraan", "wni", "berlaku", "bekasi",
        "jakarta", "bandung", "surabaya", "medan", "tangerang", "depok",
        "bogor", "semarang", "karawang", "jawa", "barat", "timur",
        "tengah", "selatan", "utara", "sulawesi", "sumatera", "kalimantan",
    }
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in lines[:15]:
        # Strip leading non-alpha noise (misal '"' sebelum nama)
        cleaned = re.sub(r"^[^A-Za-z]+", "", line).strip()
        words_raw = re.findall(r"[A-Za-z]{3,}", cleaned)
        if not (2 <= len(words_raw) <= 4):
            continue
        # Semua kata harus huruf kapital semua
        if not all(w.isupper() for w in words_raw):
            continue
        # Tidak ada angka dalam baris
        if re.search(r"\d", cleaned):
            continue
        # Tidak ada stopword
        if any(w.lower() in CAPS_STOP for w in words_raw):
            continue
        # Validasi nama dengan _is_plausible_name
        slug = "_".join(w.lower() for w in words_raw)
        if _is_plausible_name(slug):
            return slug
    return ""


def _name_after_polri(text: str) -> str:
    """
    Strategi 4 (v5.1): Ekstraksi nama pada dokumen SIM — 3 sub-strategi.

    Format SIM: Nomor SIM → Nama → Tempat/Tgl Lahir → JK → Alamat → ...
    atau dengan nomor urut: 1. Nama  2. Tempat/Tgl  3. JK  4. Alamat ...

    Sub-strategi (dijalankan berurutan, berhenti saat ada hasil):
      A. Setelah nomor SIM — toleran noise OCR (pola fleksibel 3-4-6 digit)
      B. Baris "1. NAMA" (nomor urut field SIM) — toleran noise prefix
      C. Fallback: baris pertama setelah MENGEMUDI/MENGEMUND yang lolos filter

    Perbaikan v5.7.3:
      - Sub-strategi A: toleran format "DDDD-DDDDD 00222" (spasi dalam angka)
      - Sub-strategi B: pola lebih fleksibel dengan prefix noise
      - Sub-strategi C: trigger MENGEMUND? (noise "MENGEMUND" dari foto)
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # Sub-strategi A: baris setelah nomor SIM
    # Toleran: DDDD-DDDDD-DDDDD, DDDD-DDDDD 00222, 14 digit menyatu
    SIM_NUM_PAT = re.compile(
        r"\d{3,4}[\s\-]+\d{3,6}[\s\-]*\d{3,8}"  # format 4-4-6 atau variannya
        r"|\d{10,16}",                              # atau 14 digit menyatu
    )
    for i, line in enumerate(lines):
        if SIM_NUM_PAT.search(line):
            for j in range(i + 1, min(i + 5, len(lines))):
                cand = lines[j]
                if re.match(r"^[\d\s\-\.]+$", cand):
                    continue
                name = _clean_name(cand)
                if name and _is_plausible_name(name):
                    return name

    # Sub-strategi B: baris "1. NAMA LENGKAP" (nomor urut SIM)
    for line in lines:
        m = re.match(r"^1[\s\.]+([A-Z][A-Za-z\s]{2,35})$", line.strip())
        if m:
            name = _clean_name(m.group(1))
            if name and _is_plausible_name(name):
                return name

    # Sub-strategi C: fallback — baris pertama setelah trigger MENGEMUDI
    # Skip baris yang dimulai dengan angka/simbol (nomor urut, noise)
    grab = False
    for line in lines:
        if grab:
            if re.match(r"^[\d\|\!\.\.\,\#\"\'\-\(\)\[\]]+", line):
                continue
            name = _clean_name(line)
            if name and _is_plausible_name(name):
                return name
        if re.search(r"POLR[I1]|MENGEMUDI", line.upper()):
            grab = True
    return ""


# ─── Subfungsi Ekstraksi Field Lain ──────────────────────────────────────────

def _extract_nik(text: str) -> str:
    """
    Mengekstrak NIK (Nomor Induk Kependudukan) 16 digit.

    Tiga pendekatan (v5.2 — lebih toleran OCR noise KTP):
      1. Setelah label "NIK :" — paling akurat
      2. Sequence 14-18 alfanumerik yang setelah koreksi OCR menjadi ≥15 digit
         Tangkap: "Aplb0S25010SDIUE", "321b08250103D008", "3015080501030005"
      3. Fallback sequence dengan tanda pisah (misal "351-082501030008")

    Koreksi karakter OCR: O→0, o→0, l→1, I→1, S→5, s→5, b→6, B→8,
                          Z→2, z→2, G→6, A→4, D→0

    (Rusli et al., 2020, Sec. III-C — word-to-number converter)

    Returns:
        str: NIK 16 digit, atau "" jika tidak ditemukan.
    """
    OCR_MAP = str.maketrans("OolLIisSbBzZGgAaDd", "001111558822660044")

    # Pendekatan 1: setelah label NIK
    m = re.search(
        r"\bN\s*[I1]\s*K\s*[:\-]?\s*([\dOolLIisSbBzZGgAaDd\s]{14,22})",
        text, re.IGNORECASE,
    )
    if m:
        raw    = re.sub(r"\s", "", m.group(1))
        fixed  = raw.translate(OCR_MAP)
        digits = re.sub(r"[^0-9]", "", fixed)
        if len(digits) >= MIN_NIK_DIGITS:
            return digits[:16]

    # Pendekatan 2: sequence alfanumerik 14-18 char tanpa spasi
    for candidate in re.findall(r"[0-9A-Za-z]{14,18}", re.sub(r"[\s]", "", text)):
        fixed  = candidate.translate(OCR_MAP)
        digits = re.sub(r"[^0-9]", "", fixed)
        if len(digits) >= MIN_NIK_DIGITS:
            # Validasi: 2 digit pertama adalah kode provinsi (11-99)
            if re.match(r"[1-9][0-9]", digits[:2]):
                return digits[:16]

    # Pendekatan 3: sequence dengan tanda pisah (misal "351-082501030008")
    m3 = re.search(r"(\d[\d\-\s]{13,19}\d)", text)
    if m3:
        raw    = re.sub(r"[\s\-]", "", m3.group(1))
        fixed  = raw.translate(OCR_MAP)
        digits = re.sub(r"[^0-9]", "", fixed)
        if len(digits) >= MIN_NIK_DIGITS and re.match(r"[1-9][0-9]", digits[:2]):
            return digits[:16]

    return ""


def _extract_tempat_lahir(text: str) -> str:
    """
    Mengekstrak tempat lahir dari field "Tempat/Tgl Lahir" .

    Perbaikan: toleran terhadap prefix OCR noise pada nama kota.
    Contoh: "SHEKASI-26-01-2003" → "BEKASI" (strip prefix "SH"),
            "PATENBEKASI"        → "BEKASI" (strip prefix "PATEN")

    Strategi:
      1. Cari baris label "Tempat/Tgl Lahir :" → ambil sebelum tanggal
      2. Cari pola "KATA[,/-]TGL" di mana saja → normalisasi kota
      3. Normalisasi: strip prefix noise, cocokkan ke daftar kota Indonesia

    Returns:
        str: Nama kota Title Case, atau "" jika tidak ditemukan.
    """
    # Strategi 1: label eksplisit
    for line in text.split("\n"):
        if re.search(r"TEMPAT.{0,10}(TGL|TANGGAL)?.{0,5}LAHIR", line.upper()):
            parts = re.split(r"[::]", line, maxsplit=1)
            if len(parts) == 2:
                content     = parts[1].strip()
                before_date = re.split(r"[,\-]", content)[0].strip()
                cleaned     = re.sub(r"[^A-Za-z\s]", "", before_date).strip()
                if cleaned and len(cleaned) > 2:
                    return _normalize_kota(cleaned).title()

    # Strategi 2: pola "[KOTA][,/-]DD-MM-YYYY" termasuk dengan prefix noise
    # Ambil HANYA kata TERAKHIR sebelum tanggal (nama kota = 1-2 kata terakhir)
    m = re.search(
        r"([A-Za-z][A-Za-z\s]{0,25})[,\.\-:]+\s*(\d{1,2})[-/\.](\d{2})[-/\.](\d{4})",
        text,
    )
    if m:
        raw   = re.sub(r"[^A-Za-z\s]", " ", m.group(1)).strip()
        words = [w for w in raw.split() if len(w) >= 2]
        if words:
            kota = _normalize_kota(words[-1])   # kata terakhir = nama kota
            return kota.title()

    # Strategi 3: pola parsial "KOTA-DD-MM" tanpa tahun → cari tahun di baris ±3
    lines = [l for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        m2 = re.search(
            r"([A-Za-z][A-Za-z\s]{0,20})[,\.\-:]+\s*(\d{1,2})[-/\.](\d{2})\s*$",
            line,
        )
        if m2:
            for k in range(i + 1, min(i + 4, len(lines))):
                m_yr = re.search(r"\b(19|20)\d{2}\b", lines[k])
                if m_yr:
                    raw   = re.sub(r"[^A-Za-z\s]", " ", m2.group(1)).strip()
                    words = [w for w in raw.split() if len(w) >= 2]
                    if words:
                        return _normalize_kota(words[-1]).title()
    return ""


def _normalize_kota(raw: str) -> str:
    """
    Normalisasi nama kota yang ter-OCR dengan prefix noise.
    Contoh: 'SHEKASI' → 'BEKASI',  'PATENBEKASI' → 'BEKASI'

    Strategi: coba strip prefix 1-3 karakter, cek substring matching
    ke daftar kota Indonesia yang umum muncul di KTP.
    """
    KOTA_LIST = [
        "bekasi", "jakarta", "bandung", "surabaya", "medan", "tangerang",
        "depok", "bogor", "semarang", "palembang", "makassar", "yogyakarta",
        "solo", "malang", "denpasar", "manado", "balikpapan", "samarinda",
        "pontianak", "banjarmasin", "pekanbaru", "jambi", "mataram", "ambon",
        "jayapura", "ternate", "kupang", "kendari", "karawang", "cikarang",
        "cimahi", "sukabumi", "tasikmalaya", "cirebon", "serang", "cilegon",
        "magelang", "klaten", "kudus", "jember", "kediri", "blitar", "pasuruan",
        "mojokerto", "probolinggo", "batu", "purwokerto", "tegal", "pekalongan",
        "salatiga", "madiun", "tulungagung", "banyuwangi", "situbondo",
    ]
    raw_lower = raw.lower().strip()
    if not raw_lower:
        return raw.upper()
    # 1. Match langsung
    if raw_lower in KOTA_LIST:
        return raw.upper()
    # 2. Substring: kota ada di dalam raw ("PATENBEKASI" berisi "bekasi")
    for kota in KOTA_LIST:
        if kota in raw_lower:
            return kota.upper()
    # 3. Strip prefix 1-4 karakter dari raw
    for trim in range(1, 5):
        if len(raw_lower) > trim + 2:
            candidate = raw_lower[trim:]
            if candidate in KOTA_LIST:
                return candidate.upper()
    # 4. Suffix match: akhir raw cocok dengan akhir kota (≥4 karakter)
    #    Contoh: "SHEKASI" → akhir "KASI" cocok dengan akhir "BEKASI"
    for suf_len in range(4, min(8, len(raw_lower) + 1)):
        if len(raw_lower) >= suf_len:
            suffix = raw_lower[-suf_len:]
            for kota in KOTA_LIST:
                if kota.endswith(suffix) and len(kota) >= suf_len:
                    return kota.upper()
    return raw.upper()


def _extract_tanggal_lahir(text: str) -> str:
    """
    Mengekstrak tanggal lahir. Format yang didukung: DD-MM-YYYY, DD/MM/YYYY.

    Prioritas: cari di baris "Tempat/Tgl Lahir" lebih dulu.
    Fallback: cari pola tanggal di mana saja dalam teks.

    (Rusli et al., 2020, Sec. III-C — date pattern matching)

    Returns:
        str: Tanggal dalam format DD-MM-YYYY, atau "" jika tidak ditemukan.
    """
    for line in text.split("\n"):
        if re.search(r"TEMPAT.{0,10}(TGL|TANGGAL)?.{0,5}LAHIR", line.upper()):
            m = re.search(r"(\d{1,2}[-/\s]\d{1,2}[-/\s]\d{4})", line)
            if m:
                return _normalize_date(m.group(1))

    # Fallback: pola tanggal di mana saja (termasuk setelah kota)
    m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b", text)
    if m:
        return _normalize_date(m.group(1))

    # Fallback 2: pola parsial DD-MM diikuti tahun di baris lain
    lines = [l for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        m2 = re.search(r"[,\.\-:]\s*(\d{1,2})[-/\.](\d{2})\s*$", line)
        if m2:
            for k in range(i + 1, min(i + 4, len(lines))):
                m_yr = re.search(r"\b(19|20)(\d{2})\b", lines[k])
                if m_yr:
                    dd = m2.group(1).zfill(2)
                    mm = m2.group(2)
                    yy = m_yr.group(0)
                    return f"{dd}-{mm}-{yy}"
    return ""


def _normalize_date(raw: str) -> str:
    """Normalisasi format tanggal menjadi DD-MM-YYYY."""
    cleaned = re.sub(r"[\s/]", "-", raw.strip())
    parts   = cleaned.split("-")
    if len(parts) == 3:
        d, mo, y = parts
        return f"{d.zfill(2)}-{mo.zfill(2)}-{y}"
    return raw


def _extract_jenis_kelamin(text: str) -> str:
    """
    Mengekstrak jenis kelamin. Hanya dua nilai valid yang diterima.

    (Rusli et al., 2020, Sec. III-C — binary field dengan pilihan terbatas)

    Returns:
        str: 'LAKI-LAKI', 'PEREMPUAN', atau "" jika tidak ditemukan.
    """
    upper = text.upper()
    if re.search(r"LAKI\s*[-]\s*LAKI|LAKI\s+LAKI|\bLELAKI\b|\bPRIA\b", upper):
        return "LAKI-LAKI"
    if re.search(r"PEREMPUAN|\bWANITA\b", upper):
        return "PEREMPUAN"
    return ""


def _extract_alamat(text: str) -> str:
    """
    Mengekstrak alamat dari field "Alamat :".

    Algoritma multi-baris:
      1. Temukan baris yang mengandung label "Alamat"
      2. Ambil konten setelah separator (fleksibel: ':', '"', "'", '*', '#', '@', spasi)
      3. Lanjutkan ke baris berikutnya selama bukan label field baru
      4. Berhenti setelah 3 baris atau saat menemukan label RT/RW/Kel/Kec/dll.

    Perbaikan v5.7.5:
      - Separator fleksibel: menangani kasus OCR salah baca ':' sebagai '"', "'", '*'
        Contoh dari debug: 'Alamat "Jil.Merdeka No.51...' → separator '"' bukan ':'

    Returns:
        str: Teks alamat (maks 200 karakter), atau "" jika tidak ditemukan.
    """
    lines      = text.split("\n")
    collecting = False
    addr_lines = []

    for line in lines:
        upper = line.upper().strip()
        if re.search(r"\bALAMAT\b|\bMAMAT\b", upper):  # MAMAT = OCR noise dari ALAMAT
            # v5.7.5: fleksibel separator — handle ':', '"', "'", '*', '#', '@', spasi
            # Regex: setelah kata ALAMAT, ada optional separator, lalu konten
            m = re.search(
                r'\b(?:ALAMAT|MAMAT)\b\s*[:"\u2019\u2018\'*#@\uff1a]?\s*(.+)',
                line, re.IGNORECASE
            )
            if m:
                content = m.group(1).strip()
                if content:
                    addr_lines.append(content)
            collecting = True
            continue

        if collecting:
            # Berhenti jika menemukan label field lain
            stop_pattern = (
                r"\b(RT|RW|KEL|DESA|KEC|AGAMA|PEKERJAAN|STATUS|"
                r"KEWARGANEGARAAN|BERLAKU|GOLONGAN)\b"
            )
            if re.search(stop_pattern, upper):
                break
            if line.strip():
                addr_lines.append(line.strip())
            if len(addr_lines) >= MAX_ADDR_LINES:
                break

    if addr_lines:
        combined = " ".join(addr_lines)
        return re.sub(r"\s+", " ", combined).strip()[:MAX_ADDR_LEN]

    return ""


def _extract_no_sim(text: str) -> str:
    """
    Ekstrak nomor SIM dari teks OCR.

    Format nomor SIM Indonesia:
      - Format standar : XXXX-XXXX-XXXXXX  (4-4-6 digit, misal 1223-0301-000680)
      - Format tanpa dash: XXXXXXXXXXXXXX  (14 digit menyatu)
      - Varian noise   : XXXX-XXXXX-XXXXX (4-5-5 atau 4-5-6 karena OCR spasi)
      - Varian spasi   : "3202-45094-4 00222" → 3202-4509-440022

    Nomor SIM selalu muncul di bagian ATAS dokumen, sebelum nama pemilik,
    biasanya pada baris 1–6 dokumen SIM.

    Strategi:
      1. Pola eksplisit DDDD-DDDD[D]-DDDDDD (dengan atau tanpa spasi di sekitar dash)
      2. Pola 14 digit menyatu yang dimulai dengan angka
      3. Baris bertanda nomor urut "No." di bagian atas dokumen

    Returns:
        str: Nomor SIM dalam format XXXX-XXXX-XXXXXX, atau "" jika tidak ditemukan.
    """
    OCR_MAP = str.maketrans("OolLIisSbBzZ", "001111558822")

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Strategi 1: pola DDDD-DDDD[D]-DDDDDD (format baku SIM Indonesia)
    # Toleran: spasi di sekitar dash, karakter noise OCR, part ke-2 bisa 4-6 digit
    for line in lines[:12]:
        # Bersihkan noise awal baris (pipe, kurung, dll)
        clean_line = re.sub(r"^[^\d]+", "", line)
        m = re.search(
            r"([0-9A-Za-z]{3,5})\s*[-–]\s*([0-9A-Za-z]{3,7})\s*[-–]?\s*([0-9A-Za-z]{3,8})",
            clean_line
        )
        if m:
            raw_all = m.group(1) + m.group(2) + m.group(3)
            raw_all = re.sub(r"\s", "", raw_all)  # hapus spasi jika ada
            part1 = re.sub(r"[^0-9]", "", m.group(1).translate(OCR_MAP))
            part2 = re.sub(r"[^0-9]", "", m.group(2).translate(OCR_MAP))
            part3 = re.sub(r"[^0-9]", "", m.group(3).translate(OCR_MAP))
            all_digits = part1 + part2 + part3
            if len(all_digits) >= 10 and len(part1) >= 2:
                # Pad ke format standar 4-4-6
                p1 = part1.zfill(4)[:4]
                p2 = (part2 + part3)[:4] if len(part2) < 4 else part2[:4]
                # Sisa digit setelah p1 dan p2
                rest = all_digits[len(p1):]
                p2 = rest[:4].zfill(4)
                p3 = rest[4:10].zfill(6) if len(rest) >= 6 else rest[4:].ljust(6, "0")
                if len(p3) >= 4:
                    return f"{p1}-{p2}-{p3}"

    # Strategi 2: 14 digit menyatu di baris atas (tanpa dash)
    for line in lines[:8]:
        cleaned = re.sub(r"\s+", "", line)
        fixed   = re.sub(r"[^0-9]", "", cleaned.translate(OCR_MAP))
        if len(fixed) >= 14:
            # Validasi: dimulai angka 1-9 (digit pertama tidak nol)
            if fixed[0] != "0":
                return f"{fixed[:4]}-{fixed[4:8]}-{fixed[8:14]}"

    # Strategi 3: baris "No. SIM" atau "No SIM" eksplisit
    for line in lines[:10]:
        if re.search(r"\bNo\.?\s*SIM\b", line, re.IGNORECASE):
            parts = re.split(r"[:\-]\s*", line, maxsplit=1)
            val   = parts[-1].strip() if len(parts) > 1 else ""
            fixed = re.sub(r"[^0-9]", "", val.translate(OCR_MAP))
            if len(fixed) >= 12:
                return f"{fixed[:4]}-{fixed[4:8]}-{fixed[8:14]}"

    return ""


def _extract_no_sim_voting(all_src: list) -> str:
    """Voting nomor SIM dari semua varian OCR — pilih yang paling sering muncul."""
    candidates = [_extract_no_sim(t) for t in all_src]
    candidates = [v for v in candidates if v]
    return Counter(candidates).most_common(1)[0][0] if candidates else ""


def _extract_alamat_sim(text: str, all_src: list = None) -> str:
    """
    Ekstrak alamat dari SIM.

    Struktur baris SIM (berurutan):
      1. Nomor SIM (1223-0301-000680)
      2. Nama (DIPCA ANUGRAH)
      3. Tempat, Tanggal Lahir (BEKASI, 26-01-2003)
      4. Jenis Kelamin (PRIA / WANITA)
      5. Alamat baris 1 (KP RAWA BANTENG)
      6. Alamat baris 2 (RT.05/01 CKR BARAT)
      7. Kabupaten/Kota (KAB. BEKASI)
      8. Pekerjaan (PELAJAR/MHS)
      9. Polda (METRO JAYA)

    Format nomor urut (dari foto miring) — v5.7.5:
      3. A. PRIA          <- baris jenis kelamin dengan prefix noise
      4. JL. MERDEKA NO.54
      5. RT.001/RW.004 DESA KEDIRI

    Strategi: ambil baris 5 & 6 setelah baris jenis kelamin terdeteksi.
    Fallback: cari baris yang mengandung KP/JL/DUSUN/PERUM/RT.

    Perbaikan v5.7.5:
      - _clean_sim_addr(): jangan hapus angka di akhir jika diawali NO./KM./RT./RW.
      - Strategi A: KELAMIN_PAT diperluas untuk mencocokkan "3. A. PRIA"
      - Strategi C (baru): deteksi langsung baris bernomor urut "4. JL..." / "5. RT..."
        tanpa bergantung pada baris jenis kelamin (robust untuk format foto miring)

    Returns:
        str: Alamat lengkap atau "" jika tidak ditemukan.
    """

    # Diperluas v5.7.5: cocok dengan prefix noise seperti "3. A. PRIA"
    KELAMIN_PAT = re.compile(
        r"\bPRIA\b|\bWANITA\b|\bLAKI\b|\bPEREMPUAN\b",
        re.IGNORECASE
    )
    STOP_PAT    = re.compile(
        r"\bPELAJAR\b|\bMHS\b|\bMETRO\b|\bPOLDA\b|\bSIM\b|\bBERLAKU\b|"
        r"\bSURAT\b|\bMENGEMUDI\b|\bLICENSE\b|\bDRIVING\b|\bINDONESI",
        re.IGNORECASE
    )
    ADDR_KW = re.compile(
        r"\b(?:KP|JL|JALAN|PERUM|DUSUN|GANG|GG|BLOK|RT|RW|KAV|KOMPLEK)\b",
        re.IGNORECASE
    )
    # Pola nomor urut field SIM: "4. " atau "4 " di awal baris
    NUMBERED_PAT = re.compile(r"^\s*\d+\s*[\.\)]\s*(?:[A-Z]\.\s*)?")

    def _clean_sim_addr(raw: str) -> str:
        """
        Bersihkan noise karakter dari alamat SIM: |, nomor di awal, spasi.

        Perbaikan v5.7.5: jangan hapus angka di akhir baris jika angka tersebut
        merupakan bagian dari alamat (diawali NO., KM., RT., RW., /, -).
        Contoh: "JL. MERDEKA NO.54" -> tetap "JL. MERDEKA NO.54"
                "RT.001/RW.004 DESA KEDIRI" -> tetap utuh
        """
        s = re.sub(r"\|", " ", raw)
        # Hapus nomor urut di awal baris: "5. ", "4) ", "3. A. " dll
        s = re.sub(r"^\s*\d+\s*[\.\)]\s*(?:[A-Z]\.\s*)?", "", s)
        # Hapus angka/simbol noise di akhir HANYA jika tidak diawali kata kunci alamat
        # Contoh yang TIDAK boleh dihapus: "NO.54", "KM.3", "RT.001"
        addr_suffix = re.compile(
            r"(?:NO|KM|RT|RW|KAV|BLOK|GG|GANG)\s*[\.:]?\s*\d+\s*$",
            re.IGNORECASE
        )
        if not addr_suffix.search(s):
            # Aman dihapus: angka/simbol noise yang bukan nomor jalan/RT
            s = re.sub(r"\s+[\d\.]+\s*$", "", s)
        s = re.sub(r"\b[A-Z]{1,2}\b\s*$", "", s)  # 1-2 huruf noise di akhir
        s = re.sub(r"[^\w\s/.\-]", " ", s)         # hapus simbol lain
        s = re.sub(r"\s+", " ", s).strip()
        return s

    candidates = []
    for src in (all_src or [text]):
        lines = [l.strip() for l in src.split("\n") if l.strip()]

        # ── Strategi A: 2 baris setelah baris jenis kelamin ─────────────────
        for i, line in enumerate(lines):
            if not KELAMIN_PAT.search(line):
                continue
            addr_parts = []
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j]
                if STOP_PAT.search(nxt):
                    break
                if re.search(r"KAB\.|KAB\b|KABUPATEN|KOTA\b", nxt, re.IGNORECASE):
                    break
                cleaned = _clean_sim_addr(nxt)
                if cleaned:
                    addr_parts.append(cleaned)
                if len(addr_parts) >= 2:
                    break
            if addr_parts:
                combined = " ".join(addr_parts)
                combined = re.sub(r"\s+", " ", combined).strip()
                if len(combined) >= 5:
                    candidates.append(combined)
            break

        # ── Strategi C (v5.7.5): deteksi langsung baris bernomor urut ───────
        # Untuk format foto miring: "4. JL. MERDEKA NO.54", "5. RT.001/RW.004"
        # Bekerja tanpa bergantung baris jenis kelamin.
        if not candidates:
            numbered_addr = []
            for i, line in enumerate(lines):
                # Baris harus diawali nomor urut DAN mengandung kata kunci alamat
                if not NUMBERED_PAT.match(line):
                    continue
                if not ADDR_KW.search(line):
                    continue
                if STOP_PAT.search(line):
                    continue
                cleaned = _clean_sim_addr(line)
                if not cleaned:
                    continue
                numbered_addr.append(cleaned)
                # Cek baris berikutnya: bisa lanjutan alamat (bernomor atau tidak)
                if i + 1 < len(lines):
                    nxt = lines[i + 1]
                    if (not STOP_PAT.search(nxt)
                            and not re.search(r"KAB\.|KABUPATEN|PEDAGANG|PELAJAR",
                                              nxt, re.IGNORECASE)
                            and (ADDR_KW.search(nxt) or NUMBERED_PAT.match(nxt))):
                        nxt_clean = _clean_sim_addr(nxt)
                        if nxt_clean:
                            numbered_addr.append(nxt_clean)
                if len(numbered_addr) >= 2:
                    break
            if numbered_addr:
                combined = " ".join(numbered_addr)
                combined = re.sub(r"\s+", " ", combined).strip()
                if len(combined) >= 5:
                    candidates.append(combined)

        # ── Strategi B: fallback — cari baris KP/JL/RT dll ──────────────────
        if not candidates:
            for i, line in enumerate(lines):
                if ADDR_KW.search(line) and not STOP_PAT.search(line):
                    addr_parts = [_clean_sim_addr(line)]
                    if i + 1 < len(lines):
                        nxt = lines[i + 1]
                        if (not STOP_PAT.search(nxt)
                                and not re.search(r"KAB\.|KABUPATEN", nxt, re.IGNORECASE)
                                and ADDR_KW.search(nxt)):
                            addr_parts.append(_clean_sim_addr(nxt))
                    combined = " ".join(p for p in addr_parts if p)
                    combined = re.sub(r"\s+", " ", combined).strip()
                    if len(combined) >= 5:
                        candidates.append(combined)
                    break

    if not candidates:
        return ""
    # Pilih kandidat terbersih: lebih pendek = lebih sedikit noise,
    # dengan syarat masih mengandung kata kunci alamat
    valid = [c for c in candidates if ADDR_KW.search(c)]
    pool  = valid if valid else candidates
    # Voting: yang paling sering muncul, tie-break: yang terpendek
    counted = Counter(pool)
    best_count = counted.most_common(1)[0][1]
    top = [c for c, n in counted.items() if n == best_count]
    best = min(top, key=len)
    return best[:MAX_ADDR_LEN]


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _apply_word_to_num(s: str) -> str:
    """
    Terapkan WORD_TO_NUM map untuk koreksi karakter OCR pada NIK.

    Contoh: "3216O82501O3OOO1" → "3216082501030001"
    (Rusli et al., 2020, Sec. III-C)
    """
    return "".join(WORD_TO_NUM.get(c, c) for c in s)


def _is_plausible_name(slug: str) -> bool:
    """
    Validasi apakah kandidat nama masuk akal sebagai nama orang.

    Kriteria (diperketat v5.1):
      - Panjang 1-4 kata (nama orang Indonesia umumnya 1-4 kata)
      - Total karakter minimal 3
      - Minimal 1 kata yang bukan stopword
      - Jika > 2 kata, minimal 2 kata harus bukan stopword
        (mencegah string alamat seperti "kp rawa banteng" lolos)
    """
    words     = slug.split("_")
    non_stop  = [w for w in words if w not in NAME_STOPWORDS]
    if not (MIN_NAME_WORDS <= len(words) <= MAX_NAME_WORDS and len(slug) >= MIN_NAME_LEN and len(non_stop) >= 1):
        return False
    if len(words) > 2 and len(non_stop) < 2:
        return False
    # Setiap kata harus minimal MIN_WORD_LEN huruf (menolak "ia_fe_an", "sai", dll)
    if any(len(w) < MIN_WORD_LEN for w in words):
        return False
    # Rasio vokal minimal 20%% — nama asli Indonesia selalu punya vokal
    full   = slug.replace("_", "")
    vowels = len(re.findall(r"[aeiou]", full, re.IGNORECASE))
    if full and vowels / len(full) < MIN_VOWEL_RATIO:
        return False
    return True


def _clean_name(raw: str) -> str:
    """
    Konversi string mentah menjadi slug nama lowercase_underscore.

    Proses:
      1. Hapus semua karakter non-alfabet (digit, simbol, tanda baca)
      2. Normalisasi whitespace
      3. Filter stopwords dan kata pendek (< 2 karakter)
      4. Ambil maksimal 5 kata pertama
      5. Gabungkan dengan underscore, semua lowercase

    Returns:
        str: Slug nama, atau "" jika hasilnya kosong.
    """
    cleaned = re.sub(r"[^A-Za-z\s]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 3:
        return ""
    words = [
        w for w in cleaned.split()
        if w.lower() not in NAME_STOPWORDS and len(w) >= 2
    ]
    if not words:
        return ""
    return "_".join(w.lower() for w in words[:5])


def calculate_field_confidence(all_texts: list, doc_type: str) -> dict:
    """
    Hitung confidence score setiap field dari seluruh varian OCR.

    Setiap field diekstrak secara independen dari semua 12 varian OCR.
    Confidence = frekuensi nilai terbanyak dibagi total varian.

    Args:
        all_texts (list[str]): Semua varian teks OCR (biasanya 12).
        doc_type  (str)      : Jenis dokumen untuk konteks ekstraksi.

    Returns:
        dict: {field: confidence_pct (float 0-100), ...}
    """

    if not all_texts:
        return {}

    extractors = {
        "nama"         : lambda t: _clean_name(
                             _name_from_label(t) or _name_after_nik(t) or ""),
        "nik"          : _extract_nik,
        "tempat_lahir" : _extract_tempat_lahir,
        "tanggal_lahir": _extract_tanggal_lahir,
        "jenis_kelamin": _extract_jenis_kelamin,
        "jenis_dokumen": classify_document,
    }

    n      = len(all_texts)
    result = {}
    for field, fn in extractors.items():
        vals = []
        for txt in all_texts:
            try:
                v = fn(txt)
                if v:
                    vals.append(str(v).lower().strip())
            except Exception as e:
                log.debug(f"  confidence [{field}] error: {e}")
        if not vals:
            result[field] = 0.0
        else:
            top_count     = Counter(vals).most_common(1)[0][1]
            result[field] = round(top_count / n * 100, 1)

    return result
