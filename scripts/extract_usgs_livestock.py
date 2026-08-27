"""
usgs_livestock.py

Download, cache, and extract USGS livestock water-use data into two tidy
Feather tables. Sources are the public USGS water-use archive and
ScienceBase; nothing is read from a local ad-hoc download.

    <DATA_DIR>/   holds ONLY the two cleaned tables:
        usgs_livestock_state_1960_1980.feather
        usgs_livestock_county_1985_2015.feather

    <RAW_DIR>/    cached source downloads, reused across runs
    <META_DIR>/   usgs_extraction_qa.txt, usgs_livestock_provenance.csv,
                  usgs_downloads.csv

--------------------------------------------------------------------------
DATA PROVENANCE
--------------------------------------------------------------------------
State level, 1960-1980
    ScienceBase item 584f00cee4b0260a373819db, "1950-1980 United States
    Compilation": a ZIP holding one workbook per year. Only the "LS"
    (Livestock) sheet is used. 1950 and 1955 contain no livestock category
    and are absent by design.

County level, 1985-2015
    water.usgs.gov/watuse/data/<year>/ for 1985-2010, and ScienceBase
    5af3311be4b0da30c1b245d8 for 2015. Several years are published in more
    than one container (.txt and .xls); this module prefers the delimited
    text release and falls back to Excel, sniffing the actual container
    rather than trusting the file extension.

License: USGS water-use publications are U.S. Government works in the
public domain.

--------------------------------------------------------------------------
COLUMN SELECTION
--------------------------------------------------------------------------
Livestock proper is LS / ls / LI depending on era. It is NEVER lv/LV: that
family equals livestock + aquaculture, verified exactly on all 3225 rows of
both the 1985 and 1990 files. Aquaculture (la/LA/AQ) is carried separately.

    year         gw          sw          total       consumptive
    1960-1980    LS-WGWFr    LS-WSWFr    (derived)   LS-CUsFr
    1985         ls-gwtot    ls-swtot    ls-total    ls-cuse
    1990         ls-gwtot    ls-swtot    ls-total    ls-cuse
    1995         LS-WGWFr    LS-WSWFr    LS-WFrTo    LS-CUsFr
    2000         LS-WGWFr    LS-WSWFr    LS-WFrTo    none
    2005         LS-WGWFr    LS-WSWFr    LS-WFrTo    none
    2010         LI-WGWFr    LI-WSWFr    LI-WFrTo    none
    2015         LI-WGWFr    LI-WSWFr    LI-WFrTo    none

All values are Mgal/d per the USGS data dictionaries. No unit conversion.

Two caveats that belong in the manuscript, both stated by USGS:
  - 2000 collected no consumptive use for any category, and only selected
    states compiled livestock at all; blanks are intentional.
  - The 2000 aquaculture category is not the 1995 animal-specialties
    category (it absorbed fish hatcheries from commercial). Livestock is
    unaffected by that change.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    python usgs_livestock.py
    python usgs_livestock.py --data-dir /path/to/dir --force-download

    from usgs_livestock import run
    state_df, county_df = run()
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:                                     # pragma: no cover
    requests = None

try:
    from tqdm.auto import tqdm
except ImportError:                                     # pragma: no cover
    class tqdm:                                         # minimal stand-in
        def __init__(self, iterable=None, total=None, desc=None, **kw):
            self.iterable, self.desc = iterable, desc

        def __iter__(self):
            return iter(self.iterable or ())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def update(self, n=1):
            pass

        def set_postfix_str(self, s=""):
            pass

        def close(self):
            pass

        @staticmethod
        def write(msg):
            print(msg)

warnings.filterwarnings("ignore", message="Unknown extension is not supported")

_REPO = "/scratch/hdagne1/LivestockWaterUse"

# The deliverable directory holds nothing but the two Feather tables.
DATA_DIR = f"{_REPO}/data/usgs_ls_state_county_1960_2015"

# RAW_DIR = None means the source files are downloaded into a temporary
# directory and deleted when the run finishes. Nothing is left on disk
# except the two Feather tables. Pass a path (or --raw-dir) to keep them
# cached instead, which avoids re-downloading ~19 MB on every run.
RAW_DIR = None
META_DIR = f"{_REPO}/data/_meta/usgs"

# Files an earlier version of this script wrote into DATA_DIR.
_LEGACY_SIDECARS = ("usgs_extraction_qa.txt", "usgs_livestock_provenance.csv",
                    "usgs_downloads.csv")

USGS_BASE = "https://water.usgs.gov/watuse/data"
SCIENCEBASE = "https://www.sciencebase.gov/catalog/file/get"

# Ranked candidate URLs per year. The first that downloads and parses wins.
# Delimited text is preferred over Excel: it is what was validated, needs no
# Excel engine, and avoids the second "data dictionary" sheet some .xls
# releases carry.
SOURCES: Dict[object, dict] = {
    1985: {"scale": "county", "cache": "us85co.txt",
           "urls": [f"{USGS_BASE}/1985/us85co.txt",
                    f"{USGS_BASE}/1985/us85co.xls"]},
    1990: {"scale": "county", "cache": "us90co.txt",
           "urls": [f"{USGS_BASE}/1990/us90co.txt",
                    f"{USGS_BASE}/1990/us90co.xls"]},
    1995: {"scale": "county", "cache": "us95co.txt",
           "urls": [f"{USGS_BASE}/1995/us95co.txt",
                    f"{USGS_BASE}/1995/usco1995.txt",
                    f"{USGS_BASE}/1995/us95co.xls"]},
    2000: {"scale": "county", "cache": "usco2000.txt",
           "urls": [f"{USGS_BASE}/2000/usco2000.txt",
                    f"{USGS_BASE}/2000/usco2000.xls"]},
    2005: {"scale": "county", "cache": "usco2005.xls",
           "urls": [f"{USGS_BASE}/2005/usco2005.xls"]},
    2010: {"scale": "county", "cache": "usco2010.xlsx",
           "urls": [f"{USGS_BASE}/2010/usco2010.xlsx"]},
    2015: {"scale": "county", "cache": "usco2015v2.0.csv",
           "urls": [f"{SCIENCEBASE}/5af3311be4b0da30c1b245d8"
                    f"?name=usco2015v2.0.csv"]},
    "1950-1980": {
        "scale": "state",
        "cache": "Export.1950-1980-United-States-Compilation-metadata.zip",
        "urls": [f"{SCIENCEBASE}/584f00cee4b0260a373819db?name="
                 f"Export.1950-1980-United-States-Compilation-metadata.zip"]},
}

STATE_YEARS = (1960, 1965, 1970, 1975, 1980)   # 1950/1955 have no LS sheet
STATE_SHEET, STATE_HEADER_ROW = "LS", 3

NA_TOKENS = {"", "-", "--", "---", ".", "na", "n/a", "nan", "null"}

SIG_ZIP, SIG_OLE2 = b"PK\x03\x04", b"\xD0\xCF\x11\xE0"

# ---------------------------------------------------------------------------
# per-year column contracts. Aliases are matched case-insensitively after
# stripping whitespace, so a .txt release using lowercase and a .xls release
# using mixed case both resolve.
# ---------------------------------------------------------------------------
COUNTY_SPEC: Dict[int, dict] = {
    1985: {"sheet": None,
           "state_fips": ["scode"], "county_fips": ["area"],
           "county_name": [], "year": ["year"],
           "ls_gw": ["ls-gwtot"], "ls_sw": ["ls-swtot"],
           "ls_total": ["ls-total"], "ls_cu": ["ls-cuse"],
           "aq_gw": ["la-gwtot"], "aq_sw": ["la-swtot"],
           "aq_total": ["la-total"], "aq_cu": ["la-cuse"]},
    1990: {"sheet": None,
           "state_fips": ["scode"], "county_fips": ["area"],
           "county_name": [], "year": ["year"],
           "ls_gw": ["ls-gwtot"], "ls_sw": ["ls-swtot"],
           "ls_total": ["ls-total"], "ls_cu": ["ls-cuse"],
           "aq_gw": ["la-gwtot"], "aq_sw": ["la-swtot"],
           "aq_total": ["la-total"], "aq_cu": ["la-cuse"]},
    1995: {"sheet": None,
           "state_fips": ["StateCode"], "county_fips": ["CountyCode"],
           "county_name": ["CountyName"], "year": ["Year"],
           "ls_gw": ["LS-WGWFr"], "ls_sw": ["LS-WSWFr"],
           "ls_total": ["LS-WFrTo", "LS-WTotl"], "ls_cu": ["LS-CUsFr", "LS-CUTot"],
           "aq_gw": ["LA-WGWFr"], "aq_sw": ["LA-WSWFr"],
           "aq_total": ["LA-WFrTo", "LA-WTotl"], "aq_cu": ["LA-CUsFr", "LA-CUTot"]},
    2000: {"sheet": None,
           "state_fips": ["STATEFIPS"], "county_fips": ["COUNTYFIPS"],
           "county_name": [], "year": [],
           "ls_gw": ["LS-WGWFr"], "ls_sw": ["LS-WSWFr"],
           "ls_total": ["LS-WFrTo"], "ls_cu": [],
           "aq_gw": ["LA-WGWFr"], "aq_sw": ["LA-WSWFr"],
           "aq_total": ["LA-WFrTo"], "aq_cu": []},
    2005: {"sheet": "County",
           "state_fips": ["STATEFIPS"], "county_fips": ["COUNTYFIPS"],
           "county_name": ["State-County Name"], "year": [],
           "county_name_strip_state_prefix": True,
           "ls_gw": ["LS-WGWFr"], "ls_sw": ["LS-WSWFr"],
           "ls_total": ["LS-WFrTo"], "ls_cu": [],
           "aq_gw": ["LA-WGWFr"], "aq_sw": ["LA-WSWFr"],
           "aq_total": ["LA-WFrTo"], "aq_cu": []},
    2010: {"sheet": "CountyData",
           "state_fips": ["STATEFIPS"], "county_fips": ["COUNTYFIPS"],
           "county_name": ["COUNTY"], "year": ["YEAR"],
           "ls_gw": ["LI-WGWFr"], "ls_sw": ["LI-WSWFr"],
           "ls_total": ["LI-WFrTo"], "ls_cu": [],
           "aq_gw": ["AQ-WGWFr"], "aq_sw": ["AQ-WSWFr"],
           "aq_total": ["AQ-WFrTo"], "aq_cu": []},
    2015: {"sheet": None,
           "state_fips": ["STATEFIPS"], "county_fips": ["COUNTYFIPS"],
           "county_name": ["COUNTY"], "year": ["YEAR"],
           "ls_gw": ["LI-WGWFr"], "ls_sw": ["LI-WSWFr"],
           "ls_total": ["LI-WFrTo"], "ls_cu": [],
           "aq_gw": ["AQ-WGWFr"], "aq_sw": ["AQ-WSWFr"],
           "aq_total": ["AQ-WFrTo"], "aq_cu": []},
}

STATE_FIPS: Dict[str, Tuple[str, str]] = {
    "01": ("Alabama", "AL"), "02": ("Alaska", "AK"), "04": ("Arizona", "AZ"),
    "05": ("Arkansas", "AR"), "06": ("California", "CA"), "08": ("Colorado", "CO"),
    "09": ("Connecticut", "CT"), "10": ("Delaware", "DE"),
    "11": ("District of Columbia", "DC"), "12": ("Florida", "FL"),
    "13": ("Georgia", "GA"), "15": ("Hawaii", "HI"), "16": ("Idaho", "ID"),
    "17": ("Illinois", "IL"), "18": ("Indiana", "IN"), "19": ("Iowa", "IA"),
    "20": ("Kansas", "KS"), "21": ("Kentucky", "KY"), "22": ("Louisiana", "LA"),
    "23": ("Maine", "ME"), "24": ("Maryland", "MD"), "25": ("Massachusetts", "MA"),
    "26": ("Michigan", "MI"), "27": ("Minnesota", "MN"), "28": ("Mississippi", "MS"),
    "29": ("Missouri", "MO"), "30": ("Montana", "MT"), "31": ("Nebraska", "NE"),
    "32": ("Nevada", "NV"), "33": ("New Hampshire", "NH"), "34": ("New Jersey", "NJ"),
    "35": ("New Mexico", "NM"), "36": ("New York", "NY"), "37": ("North Carolina", "NC"),
    "38": ("North Dakota", "ND"), "39": ("Ohio", "OH"), "40": ("Oklahoma", "OK"),
    "41": ("Oregon", "OR"), "42": ("Pennsylvania", "PA"), "44": ("Rhode Island", "RI"),
    "45": ("South Carolina", "SC"), "46": ("South Dakota", "SD"),
    "47": ("Tennessee", "TN"), "48": ("Texas", "TX"), "49": ("Utah", "UT"),
    "50": ("Vermont", "VT"), "51": ("Virginia", "VA"), "53": ("Washington", "WA"),
    "54": ("West Virginia", "WV"), "55": ("Wisconsin", "WI"), "56": ("Wyoming", "WY"),
    "60": ("American Samoa", "AS"), "66": ("Guam", "GU"),
    "69": ("Northern Mariana Islands", "MP"), "72": ("Puerto Rico", "PR"),
    "78": ("U.S. Virgin Islands", "VI"),
}

# Conterminous US: 48 states + DC. Excludes Alaska (02), Hawaii (15),
# Puerto Rico (72), U.S. Virgin Islands (78) and the Pacific territories.
NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_FIPS = {f for f in
              ("01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
               "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
               "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
               "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
               "47", "48", "49", "50", "51", "53", "54", "55", "56")}

STATE_SCHEMA = [
    "year", "state_fips", "state_abbrev", "state_name",
    "ls_gw_fresh_mgd", "ls_sw_fresh_mgd", "ls_withdrawal_fresh_mgd",
    "ls_consumptive_fresh_mgd", "ls_withdrawal_is_derived", "source_file",
]

# What actually gets written to disk. Working columns used during
# extraction (year_reported, county_name_source, source_file, the
# aquaculture block, the derived-total flag, the split FIPS parts) are
# dropped: they are diagnostics, and they all remain in the QA report and
# the provenance CSV.
#
# `fips` is kept deliberately. It is the only stable join key to county
# geometries, Census tables and NASS inventories; state_fips is its first
# two characters, so nothing is lost by dropping the split columns.
STATE_OUTPUT = [
    "year", "state_fips", "state_abbrev", "state_name",
    "ls_gw_fresh_mgd", "ls_sw_fresh_mgd", "ls_withdrawal_fresh_mgd",
    "ls_consumptive_fresh_mgd", "ls_cw_ratio",
]

COUNTY_OUTPUT = [
    "year", "fips", "state_abbrev", "state_name", "county_name",
    "ls_gw_fresh_mgd", "ls_sw_fresh_mgd", "ls_withdrawal_fresh_mgd",
    "ls_consumptive_fresh_mgd", "ls_cw_ratio",
]

# Text columns pinned to a nullable string dtype so the Feather schema is
# identical regardless of pandas version, and so an all-NA county_name
# (1985, 1990, 2000) does not silently land as float64 or object.
TEXT_COLUMNS = ("fips", "state_fips", "county_fips", "state_abbrev",
                "state_name", "county_name", "county_name_source",
                "source_file")

COUNTY_SCHEMA = [
    "year", "year_reported", "fips", "state_fips", "county_fips",
    "state_abbrev", "state_name", "county_name", "county_name_source",
    "ls_gw_fresh_mgd", "ls_sw_fresh_mgd", "ls_withdrawal_fresh_mgd",
    "ls_consumptive_fresh_mgd",
    "aq_gw_fresh_mgd", "aq_sw_fresh_mgd", "aq_withdrawal_fresh_mgd",
    "aq_consumptive_fresh_mgd",
    "ls_withdrawal_is_derived", "source_file",
]


class Report:
    """
    Writes the full narrative to the QA file. The console stays quiet unless
    echo=True, so a notebook shows progress bars instead of hundreds of
    lines. Nothing is lost: the file always gets everything.
    """

    def __init__(self, path: Optional[Path] = None, echo: bool = False):
        # path=None keeps the narrative entirely in memory, so no QA file
        # is written. Every rep() call still works; nothing needs guarding.
        self.handle = open(path, "w", encoding="utf-8") if path else None
        self.echo = echo

    def __call__(self, text: str = "") -> None:
        if self.echo:
            tqdm.write(text)
        if self.handle is not None:
            self.handle.write(text + "\n")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


class ExtractionError(RuntimeError):
    """Raised when a year cannot be parsed as specified. Never swallowed."""


# ---------------------------------------------------------------------------
# download and cache
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rep: Report, key, spec: dict, raw_dir: Path,
          force: bool, timeout: int) -> Tuple[Optional[Path], dict]:
    """
    Download the first URL that works, into raw_dir/<cache name>.
    A cached file is reused unless --force-download. Returns (path, meta).
    """
    dest = raw_dir / spec["cache"]
    meta = {"key": str(key), "spatial_scale": spec["scale"],
            "cache_file": spec["cache"], "url": None,
            "bytes": None, "sha256": None, "retrieved_utc": None,
            "status": None}

    if dest.exists() and not force:
        meta.update(url="(cached)", bytes=dest.stat().st_size,
                    sha256=sha256_of(dest), status="cached",
                    retrieved_utc=dt.datetime.fromtimestamp(
                        dest.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds"))
        rep(f"  {key}: cached {dest.name} "
            f"({meta['bytes']:,} B, sha256 {meta['sha256'][:12]}...)")
        return dest, meta

    if requests is None:
        rep(f"  {key}: requests is not installed and no cached copy exists")
        meta["status"] = "no requests module"
        return None, meta

    for url in spec["urls"]:
        rep(f"  {key}: GET {url}")
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "LivestockWaterUse/1.0"})
            resp.raise_for_status()
        except Exception as err:
            rep(f"       failed: {type(err).__name__}: {err}")
            continue

        content = resp.content
        # An HTML error page returned with status 200 is a real failure mode
        # on this archive; reject it rather than trying to parse it.
        head = content[:512].lstrip().lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            rep("       failed: server returned an HTML page, not a data file")
            continue

        raw_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        meta.update(url=url, bytes=len(content), sha256=sha256_of(dest),
                    status="downloaded",
                    retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        rep(f"       ok: {len(content):,} B, sha256 {meta['sha256'][:12]}...")
        return dest, meta

    rep(f"  {key}: ALL candidate URLs failed")
    meta["status"] = "all urls failed"
    return None, meta


# ---------------------------------------------------------------------------
# generic readers
# ---------------------------------------------------------------------------

def container_of(path: Path) -> str:
    head = path.open("rb").read(8)
    if head.startswith(SIG_ZIP):
        return "xlsx"
    if head.startswith(SIG_OLE2):
        return "xls"
    return "text"


def decode_normalized(path: Path) -> str:
    """utf-8 then latin-1; CR and CRLF collapse to LF (1995 is CR-only)."""
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ExtractionError(f"{path.name}: cannot decode")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_tabular_text(path: Path) -> pd.DataFrame:
    """
    Delimiter-agnostic reader for the USGS text releases. Detects tab vs
    comma by field-count consistency, skips any leading title or citation
    line, and trims a header line carrying extra trailing delimiters.
    """
    lines = [ln for ln in decode_normalized(path).split("\n") if ln.strip()]
    if len(lines) < 2:
        raise ExtractionError(f"{path.name}: fewer than two non-blank lines")

    sample = lines[:25]
    best, best_score = None, -1.0
    for sep in ("\t", ","):
        counts = [ln.count(sep) for ln in sample]
        if max(counts) == 0:
            continue
        modal = max(set(counts), key=counts.count)
        score = counts.count(modal) / len(counts) + modal / 1000.0
        if score > best_score:
            best, best_score = sep, score
    if best is None:
        raise ExtractionError(f"{path.name}: no tab or comma delimiter found")

    # Choose the header line by scoring, not by field count. A header is
    # full, textual, and unique; a citation line (2015) puts everything in
    # field 0; a data line is mostly numeric. Width alone cannot separate
    # them: the 1995 header is WIDER than its data rows (trailing tab), and
    # so is the 2015 citation line (commas inside the quoted title).
    def header_score(line: str) -> float:
        fields = [f.strip() for f in line.split(best)]
        if not fields:
            return -1.0
        filled = [f for f in fields if f]
        if len(filled) < 2:
            return -1.0
        textual = sum(1 for f in filled
                      if not re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", f))
        return (len(filled) / len(fields)
                + 2.0 * textual / len(filled)
                + len(set(filled)) / len(filled))

    start = max(range(min(5, len(lines) - 1)), key=lambda i: header_score(lines[i]))

    names = lines[start].split(best)
    n_data = lines[start + 1].count(best) + 1
    while len(names) > n_data and not names[-1].strip():
        names.pop()                      # 1995: trailing tab on the header
    # Any remaining collision would make pandas reject the read outright.
    seen: Dict[str, int] = {}
    for i, nm in enumerate(names):
        key = nm.strip()
        if key in seen:
            seen[key] += 1
            names[i] = f"{key}.{seen[key]}"
        else:
            seen[key] = 0

    return pd.read_csv(io.StringIO("\n".join(lines[start + 1:])), sep=best,
                       header=None, names=names, dtype=str,
                       engine="python", on_bad_lines="warn")


def read_sheet(path: Path, sheet: Optional[str], header_row: int = 0) -> pd.DataFrame:
    engine = "openpyxl" if container_of(path) == "xlsx" else "xlrd"
    try:
        book = pd.ExcelFile(path, engine=engine)
    except ImportError as err:
        raise ExtractionError(f"{path.name}: {err} (pip install openpyxl xlrd)")
    target = sheet if sheet in book.sheet_names else book.sheet_names[0]
    df = pd.read_excel(path, sheet_name=target, skiprows=header_row,
                       engine=engine, dtype=str)
    book.close()
    return df


def load_source(path: Path, sheet: Optional[str]) -> pd.DataFrame:
    kind = container_of(path)
    if kind == "text":
        return read_tabular_text(path)
    return read_sheet(path, sheet)


# ---------------------------------------------------------------------------
# column resolution and coercion
# ---------------------------------------------------------------------------

def resolve(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    """Case- and whitespace-insensitive column lookup over an alias list."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        hit = lookup.get(alias.strip().lower())
        if hit is not None:
            return hit
    return None


def to_number(series: pd.Series) -> pd.Series:
    """Text to float. USGS NA tokens become NaN, never zero."""
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower().isin(NA_TOKENS))
    return pd.to_numeric(s, errors="coerce")


def pad_fips(series: pd.Series, width: int) -> pd.Series:
    """Fixed-width zero-padded FIPS. Repairs int-parsed codes (1001 -> 01001)."""
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.mask(s.str.lower().isin(NA_TOKENS))
    return s.str.zfill(width)


def numeric_or_nan(df: pd.DataFrame, aliases: Sequence[str],
                   n: int) -> Tuple[pd.Series, Optional[str]]:
    """Resolve and coerce, or return an all-NaN column when not reported."""
    col = resolve(df, aliases) if aliases else None
    if col is None:
        return pd.Series(np.nan, index=range(n), dtype="float64"), None
    return to_number(df[col]).reset_index(drop=True), col


# ---------------------------------------------------------------------------
# state extraction
# ---------------------------------------------------------------------------

def extract_state(rep: Report, zip_path: Path) -> Tuple[pd.DataFrame, List[dict]]:
    rep("\n" + "=" * 78)
    rep("STATE-LEVEL 1960-1980")
    rep("=" * 78)
    frames, prov = [], []

    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        bar = tqdm(STATE_YEARS, desc="State-Level  1960-1980", unit="yr")
        for year in bar:
            bar.set_postfix_str(str(year))
            name = f"Export.{year}-United-States-Compilation.xlsx"
            match = next((m for m in members if m.endswith(name)), None)
            if match is None:
                raise ExtractionError(
                    f"{year}: '{name}' not found in {zip_path.name}. "
                    f"Archive holds: {members[:12]}")

            with z.open(match) as fh:
                payload = fh.read()
            try:
                book = pd.ExcelFile(io.BytesIO(payload), engine="openpyxl")
            except ImportError as err:
                raise ExtractionError(
                    f"{year}: {err}\n"
                    f"      Install the Excel engines into the active "
                    f"environment:  pip install openpyxl xlrd") from None
            if STATE_SHEET not in book.sheet_names:
                raise ExtractionError(
                    f"{year}: no '{STATE_SHEET}' sheet; sheets are {book.sheet_names}")
            raw = book.parse(STATE_SHEET, skiprows=STATE_HEADER_ROW, dtype=str)
            book.close()

            for needed in ("Area", "LS-WGWFr", "LS-WSWFr", "LS-CUsFr"):
                if resolve(raw, [needed]) is None:
                    raise ExtractionError(
                        f"{year}: LS sheet is missing '{needed}'. "
                        f"Found {list(raw.columns)}")

            out = pd.DataFrame()
            out["state_fips"] = pad_fips(raw[resolve(raw, ["Area"])], 2)
            out["ls_gw_fresh_mgd"] = to_number(raw[resolve(raw, ["LS-WGWFr"])])
            out["ls_sw_fresh_mgd"] = to_number(raw[resolve(raw, ["LS-WSWFr"])])
            out["ls_consumptive_fresh_mgd"] = to_number(raw[resolve(raw, ["LS-CUsFr"])])
            # State releases report no withdrawal total; it is gw + sw. If
            # either part is missing the total is missing, not a partial sum.
            out["ls_withdrawal_fresh_mgd"] = (out["ls_gw_fresh_mgd"]
                                              + out["ls_sw_fresh_mgd"])
            out["ls_withdrawal_is_derived"] = True

            before = len(out)
            out = out[out["state_fips"].isin(STATE_FIPS)].copy()
            out["year"] = np.int16(year)
            out["source_file"] = name
            out["state_name"] = out["state_fips"].map(lambda f: STATE_FIPS[f][0])
            out["state_abbrev"] = out["state_fips"].map(lambda f: STATE_FIPS[f][1])

            rep(f"  {year}: {len(out)} areas"
                + (f" ({before - len(out)} unmappable dropped)" if before != len(out) else "")
                + f" | CU non-null {int(out['ls_consumptive_fresh_mgd'].notna().sum())}")

            for role, col in (("gw", "LS-WGWFr"), ("sw", "LS-WSWFr"),
                              ("withdrawal_total", "(derived: LS-WGWFr + LS-WSWFr)"),
                              ("consumptive", "LS-CUsFr")):
                prov.append({"year": year, "spatial_scale": "state",
                             "source_file": name, "role": role,
                             "source_column": col})
            frames.append(out[STATE_SCHEMA])
        bar.close()

    return pd.concat(frames, ignore_index=True), prov


# ---------------------------------------------------------------------------
# county extraction, one spec-driven path for every year
# ---------------------------------------------------------------------------

def extract_county_year(rep: Report, path: Path, year: int) -> Tuple[pd.DataFrame, List[dict]]:
    spec = COUNTY_SPEC[year]
    raw = load_source(path, spec.get("sheet"))
    raw.columns = [str(c).strip() for c in raw.columns]   # 'DO-TOTAL ' etc.
    n = len(raw)

    sf = resolve(raw, spec["state_fips"])
    cf = resolve(raw, spec["county_fips"])
    if sf is None or cf is None:
        raise ExtractionError(
            f"{year}: cannot resolve FIPS columns "
            f"{spec['state_fips']} / {spec['county_fips']} in {path.name}")

    out = pd.DataFrame(index=range(n))
    out["state_fips"] = pad_fips(raw[sf], 2).reset_index(drop=True)
    out["county_fips"] = pad_fips(raw[cf], 3).reset_index(drop=True)

    cn = resolve(raw, spec["county_name"]) if spec["county_name"] else None
    if cn is None:
        # 1985, 1990 and 2000 carry no county-name column. Left empty here;
        # backfilled from a FIPS crosswalk after all years are loaded.
        out["county_name"] = pd.NA
        out["county_name_source"] = pd.NA
    else:
        names = raw[cn].astype(str).str.strip()
        if spec.get("county_name_strip_state_prefix"):
            # "Alabama-Autauga County" -> "Autauga County". Split on the FIRST
            # hyphen only; state names contain none, county names can
            # (Miami-Dade).
            names = names.str.split("-", n=1).str[-1].str.strip()
        out["county_name"] = names.reset_index(drop=True)
        out["county_name_source"] = f"{year} source file"

    yc = resolve(raw, spec["year"]) if spec["year"] else None
    out["year_reported"] = (to_number(raw[yc]).reset_index(drop=True)
                            if yc else pd.Series(pd.NA, index=range(n)))

    used: Dict[str, Optional[str]] = {}
    for target, key in (("ls_gw_fresh_mgd", "ls_gw"),
                        ("ls_sw_fresh_mgd", "ls_sw"),
                        ("ls_withdrawal_fresh_mgd", "ls_total"),
                        ("ls_consumptive_fresh_mgd", "ls_cu"),
                        ("aq_gw_fresh_mgd", "aq_gw"),
                        ("aq_sw_fresh_mgd", "aq_sw"),
                        ("aq_withdrawal_fresh_mgd", "aq_total"),
                        ("aq_consumptive_fresh_mgd", "aq_cu")):
        out[target], used[key] = numeric_or_nan(raw, spec[key], n)

    # A livestock column the spec says must exist but does not resolve is a
    # hard failure: the alternative is silently emitting an empty series.
    for key in ("ls_gw", "ls_sw", "ls_total"):
        if spec[key] and used[key] is None:
            raise ExtractionError(
                f"{year}: expected one of {spec[key]} in {path.name}; "
                f"found {list(raw.columns)[:20]} ...")

    before = len(out)
    out = out[out["state_fips"].isin(STATE_FIPS)
              & out["county_fips"].str.fullmatch(r"\d{3}").fillna(False)].copy()
    if before != len(out):
        rep(f"  {year}: dropped {before - len(out)} rows without a valid FIPS "
            f"(trailing notes or blank rows in the source)")

    out["fips"] = out["state_fips"] + out["county_fips"]
    out["year"] = np.int16(year)
    out["source_file"] = path.name
    out["ls_withdrawal_is_derived"] = False
    out["state_name"] = out["state_fips"].map(lambda f: STATE_FIPS[f][0])
    out["state_abbrev"] = out["state_fips"].map(lambda f: STATE_FIPS[f][1])

    prov = [{"year": year, "spatial_scale": "county", "source_file": path.name,
             "role": role, "source_column": used[key] or "(not reported)"}
            for role, key in (("gw", "ls_gw"), ("sw", "ls_sw"),
                              ("withdrawal_total", "ls_total"),
                              ("consumptive", "ls_cu"),
                              ("aq_gw", "aq_gw"), ("aq_sw", "aq_sw"),
                              ("aq_withdrawal_total", "aq_total"),
                              ("aq_consumptive", "aq_cu"))]

    rep(f"  {year}: {len(out)} counties | "
        f"W non-null {int(out['ls_withdrawal_fresh_mgd'].notna().sum())} | "
        f"CU non-null {int(out['ls_consumptive_fresh_mgd'].notna().sum())} | "
        f"cols {used['ls_gw']}, {used['ls_sw']}, {used['ls_total']}, "
        f"{used['ls_cu'] or 'no CU'}")
    return out[COUNTY_SCHEMA], prov


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def qa_state(rep: Report, df: pd.DataFrame) -> None:
    rep("\n" + "=" * 78)
    rep("QA -- STATE TABLE")
    rep("=" * 78)
    rep(f"\n  {'year':<6}{'areas':>7}{'dup':>6}{'W nn':>7}{'CU nn':>7}"
        f"{'natl W':>13}{'natl CU':>13}")
    for year, g in df.groupby("year"):
        rep(f"  {year:<6}{len(g):>7}{int(g['state_fips'].duplicated().sum()):>6}"
            f"{int(g['ls_withdrawal_fresh_mgd'].notna().sum()):>7}"
            f"{int(g['ls_consumptive_fresh_mgd'].notna().sum()):>7}"
            f"{g['ls_withdrawal_fresh_mgd'].sum(min_count=1):>13,.1f}"
            f"{g['ls_consumptive_fresh_mgd'].sum(min_count=1):>13,.1f}")

    rep("\n  consumption-to-withdrawal ratio")
    for year, g in df.groupby("year"):
        w, cu = g["ls_withdrawal_fresh_mgd"], g["ls_consumptive_fresh_mgd"]
        ok = w.notna() & cu.notna() & (w > 0)
        if not ok.any():
            continue
        r = (cu / w)[ok]
        rep(f"  {year}: n={int(ok.sum())} min={r.min():.3f} "
            f"median={r.median():.3f} max={r.max():.3f} "
            f"| above 1.0: {int((r > 1.0).sum())}")
    rep("\n  Note: state values are rounded to 2 significant digits "
        "(3 for 1980), so a ratio marginally above 1.0 is a rounding artifact.")


def qa_county(rep: Report, df: pd.DataFrame) -> None:
    rep("\n" + "=" * 78)
    rep("QA -- COUNTY TABLE")
    rep("=" * 78)

    rep("\n  identifiers and coverage")
    rep(f"  {'year':<6}{'rows':>7}{'uniq':>7}{'dup':>5}{'bad':>5}{'states':>8}"
        f"{'names':>7}{'W nn':>7}{'CU nn':>7}")
    for year, g in df.groupby("year"):
        bad = int((~g["fips"].astype(str).str.fullmatch(r"\d{5}").fillna(False)).sum())
        rep(f"  {year:<6}{len(g):>7}{g['fips'].nunique():>7}"
            f"{int(g['fips'].duplicated().sum()):>5}{bad:>5}"
            f"{g['state_fips'].nunique():>8}"
            f"{int(g['county_name'].notna().sum()):>7}"
            f"{int(g['ls_withdrawal_fresh_mgd'].notna().sum()):>7}"
            f"{int(g['ls_consumptive_fresh_mgd'].notna().sum()):>7}")

    rep("\n  internal consistency: reported total vs gw + sw")
    for year, g in df.groupby("year"):
        gw, sw, tot = (g["ls_gw_fresh_mgd"], g["ls_sw_fresh_mgd"],
                       g["ls_withdrawal_fresh_mgd"])
        ok = gw.notna() & sw.notna() & tot.notna()
        if not ok.any():
            rep(f"  {year}: no comparable rows")
            continue
        dev = (gw + sw - tot).abs()[ok]
        rep(f"  {year}: within 0.005 on {int((dev <= 0.005).sum())}/{int(ok.sum())}"
            f" rows (max deviation {dev.max():.4g})")

    rep("\n  national totals, Mgal/d")
    rep(f"  {'year':<6}{'livestock W':>14}{'aquaculture W':>16}{'livestock CU':>15}")
    for year, g in df.groupby("year"):
        cu = g["ls_consumptive_fresh_mgd"].sum(min_count=1)
        rep(f"  {year:<6}{g['ls_withdrawal_fresh_mgd'].sum(min_count=1):>14,.1f}"
            f"{g['aq_withdrawal_fresh_mgd'].sum(min_count=1):>16,.1f}"
            + (f"{cu:>15,.1f}" if pd.notna(cu) else f"{'not reported':>15}"))

    rep("\n  physical plausibility: consumptive use above withdrawal")
    for year, g in df.groupby("year"):
        w, cu = g["ls_withdrawal_fresh_mgd"], g["ls_consumptive_fresh_mgd"]
        ok = w.notna() & cu.notna() & (w > 0)
        if not ok.any():
            continue
        ratio = (cu / w)[ok]
        over = ratio > 1.0
        rep(f"  {year}: n={int(ok.sum())} median={ratio.median():.3f} "
            f"max={ratio.max():.3f} | above 1.0: {int(over.sum())}")
        for idx in ratio[over].nlargest(5).index:
            r = g.loc[idx]
            rep(f"      {r['fips']} {r['state_abbrev']} "
                f"{str(r['county_name'])[:24]:<24} "
                f"W={r['ls_withdrawal_fresh_mgd']:g} "
                f"CU={r['ls_consumptive_fresh_mgd']:g} "
                f"ratio={r['ls_consumptive_fresh_mgd'] / r['ls_withdrawal_fresh_mgd']:.2f}")

    rep("\n  reported year vs survey year")
    for year, g in df.groupby("year"):
        yr = g["year_reported"].dropna()
        if yr.empty:
            rep(f"  {year}: source has no year column")
        else:
            vc = {int(k): int(v) for k, v in yr.value_counts().items()}
            flag = "   <-- MISMATCH" if len(vc) > 1 or list(vc) != [year] else ""
            rep(f"  {year}: {vc}{flag}")

    rep("\n  county-name provenance")
    for year, g in df.groupby("year"):
        vc = g["county_name_source"].value_counts(dropna=False).to_dict()
        vc = {("(unresolved)" if pd.isna(k) else k): int(v) for k, v in vc.items()}
        rep(f"  {year}: {vc}")

    rep("\n  FIPS coverage across years (county boundaries change over time)")
    sets = {int(y): set(g["fips"]) for y, g in df.groupby("year")}
    everywhere = set.intersection(*sets.values()) if sets else set()
    rep(f"  present in all {len(sets)} years: {len(everywhere)}")
    for year in sorted(sets):
        only_missing = everywhere - sets[year]
        extra = sets[year] - set.union(*(v for k, v in sets.items() if k != year))
        rep(f"  {year}: {len(sets[year])} FIPS"
            f" | unique to this year: {len(extra)}"
            + (f" {sorted(extra)[:10]}" if extra else ""))

    rep("\n  negative values")
    val_cols = [c for c in df.columns if c.endswith("_mgd")]
    hits = {c: int((df[c] < 0).sum()) for c in val_cols}
    bad = {c: v for c, v in hits.items() if v}
    rep(f"    {bad if bad else 'none'}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def restrict_to_conus(rep: Report, df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep the 48 conterminous states plus DC."""
    fips_col = "state_fips" if "state_fips" in df.columns else None
    key = df[fips_col] if fips_col else df["fips"].str[:2]
    keep = key.isin(CONUS_FIPS)
    removed = df.loc[~keep]

    rep("\n" + "=" * 78)
    rep(f"CONUS RESTRICTION -- {label}")
    rep("=" * 78)
    rep(f"  kept {int(keep.sum()):,} rows, removed {int((~keep).sum()):,}")
    if not removed.empty:
        by_state = (removed.groupby("state_abbrev", observed=True).size()
                    .sort_values(ascending=False))
        rep(f"  removed by state: {dict(by_state)}")
        rep(f"  distinct CONUS states retained: {key[keep].nunique()} "
            f"(expected 49: 48 states + DC)")
    return df.loc[keep].reset_index(drop=True)


def add_cw_ratio(rep: Report, df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Consumption-to-withdrawal ratio, livestock only.

    Numerator and denominator come from the SAME USGS category and the SAME
    water type in every year: LS/ls consumptive use over LS/ls freshwater
    withdrawal. This is the check that matters. Dividing a livestock
    consumptive use by an all-category withdrawal (TO-*), or by a different
    livestock variant (LV-*), would produce a number that looks reasonable
    and means nothing.

    Guards:
      - withdrawal <= 0  ->  NaN, never inf and never a fabricated value.
      - consumptive use not reported (2000 onward) -> NaN for the whole year.
      - ratios above 1.0 are left exactly as reported. They are physically
        impossible and almost always an artifact of two-significant-figure
        rounding at small withdrawals; capping them would hide a property
        of the source data that belongs in the Methods section.
    """
    w = df["ls_withdrawal_fresh_mgd"]
    cu = df["ls_consumptive_fresh_mgd"]
    df["ls_cw_ratio"] = cu / w.where(w > 0)

    rep("\n" + "=" * 78)
    rep(f"CONSUMPTION-TO-WITHDRAWAL RATIO -- {label}")
    rep("=" * 78)
    rep("  numerator   ls_consumptive_fresh_mgd   (USGS LS/ls consumptive use, fresh)")
    rep("  denominator ls_withdrawal_fresh_mgd    (USGS LS/ls withdrawal, fresh)")
    rep("  same category and same water type on both sides, every year.")
    rep(f"\n  {'year':<6}{'n':>6}{'zero W':>8}{'no CU':>7}{'min':>8}{'p25':>8}"
        f"{'median':>8}{'p75':>8}{'max':>8}{'=1.00':>7}{'>1.00':>7}")
    for year, g in df.groupby("year"):
        r = g["ls_cw_ratio"].dropna()
        zero_w = int((g["ls_withdrawal_fresh_mgd"] <= 0).sum())
        no_cu = int(g["ls_consumptive_fresh_mgd"].isna().sum())
        if r.empty:
            rep(f"  {year:<6}{0:>6}{zero_w:>8}{no_cu:>7}"
                f"{'  consumptive use not reported this year':>50}")
            continue
        rep(f"  {year:<6}{len(r):>6}{zero_w:>8}{no_cu:>7}"
            f"{r.min():>8.3f}{r.quantile(.25):>8.3f}{r.median():>8.3f}"
            f"{r.quantile(.75):>8.3f}{r.max():>8.3f}"
            f"{int(((r - 1.0).abs() <= 1e-9).sum()):>7}"
            f"{int((r > 1.0).sum()):>7}")

    r_all = df["ls_cw_ratio"].dropna()
    if not r_all.empty:
        rep("\n  the ratio is not continuous: USGS set consumptive use equal to")
        rep("  withdrawal for a large share of rows, and two-significant-figure")
        rep("  reporting quantises the rest. Ten most frequent values:")
        top = r_all.round(6).value_counts().head(10)
        for val, n in top.items():
            rep(f"    {val:>8.4f}  {n:>6,} rows  ({100 * n / len(r_all):>5.1f}%)")
        rep(f"    exactly 1.0000: {int((r_all == 1.0).sum()):,} rows "
            f"({100 * (r_all == 1.0).mean():.1f}%)")
        rep(f"    exactly 0.0000: {int((r_all == 0.0).sum()):,} rows "
            f"({100 * (r_all == 0.0).mean():.1f}%)")
        rep(f"    above 1.0:      {int((r_all > 1.0).sum()):,} rows "
            f"({100 * (r_all > 1.0).mean():.1f}%)")

    rep("\n  volume-weighted ratio, sum(CU) / sum(W) -- use this for aggregates,")
    rep("  not the mean of the row ratios, which over-weights tiny counties.")
    for year, g in df.groupby("year"):
        tw = g["ls_withdrawal_fresh_mgd"].sum(min_count=1)
        tc = g["ls_consumptive_fresh_mgd"].sum(min_count=1)
        if pd.isna(tc) or pd.isna(tw) or tw <= 0:
            rep(f"  {year}: not computable (consumptive use not reported)")
        else:
            rep(f"  {year}: {tc / tw:.4f}   (CU {tc:,.1f} / W {tw:,.1f} Mgal/d)")
    return df


def backfill_county_names(rep: Report, df: pd.DataFrame) -> pd.DataFrame:
    """
    1985, 1990 and 2000 have no county-name column in the source. Fill from
    a FIPS crosswalk built out of the years that do (1995, 2005, 2010,
    2015), preferring the year nearest in time to the target.

    County FIPS codes are not stable across three decades: codes are
    created, merged and renumbered (Dade -> Miami-Dade in 1997, Shannon ->
    Oglala Lakota in 2015, several Virginia independent cities merged into
    their surrounding counties, Alaska boroughs reorganized repeatedly).
    Any code with no later counterpart stays empty rather than being
    matched to a neighbouring county. Every filled name records its origin
    in county_name_source.
    """
    rep("\n" + "=" * 78)
    rep("COUNTY NAME BACKFILL")
    rep("=" * 78)

    named = df[df["county_name"].notna()]
    donors = sorted(named["year"].unique())
    if not donors:
        rep("  no year carries county names; nothing to backfill")
        return df
    rep(f"  donor years (names present in source): {[int(y) for y in donors]}")

    lookup = {int(y): dict(zip(g["fips"], g["county_name"]))
              for y, g in named.groupby("year")}

    for target in sorted(df["year"].unique()):
        mask = (df["year"] == target) & df["county_name"].isna()
        n_missing = int(mask.sum())
        if n_missing == 0:
            continue
        fips = df.loc[mask, "fips"]
        filled = pd.Series(pd.NA, index=fips.index, dtype="object")
        origin = pd.Series(pd.NA, index=fips.index, dtype="object")

        for donor in sorted(lookup, key=lambda d: (abs(d - int(target)), d)):
            todo = filled.isna()
            if not todo.any():
                break
            hit = fips[todo].map(lookup[donor])
            got = hit.notna()
            if got.any():
                filled.loc[hit.index[got]] = hit[got]
                origin.loc[hit.index[got]] = f"{donor} crosswalk"

        df.loc[mask, "county_name"] = filled
        df.loc[mask, "county_name_source"] = origin

        still = int(filled.isna().sum())
        by_donor = origin.dropna().value_counts().to_dict()
        rep(f"  {int(target)}: {n_missing - still}/{n_missing} filled "
            f"{by_donor}")
        if still:
            unresolved = df.loc[mask & df["county_name"].isna(),
                                ["fips", "state_abbrev"]]
            rep(f"      {still} FIPS with no counterpart in any donor year "
                f"(county created, merged or renumbered since): "
                f"{sorted(unresolved['fips'].tolist())[:20]}"
                + (" ..." if still > 20 else ""))
    return df


def migrate_legacy(base: Path, raw_dir: Path, meta_dir: Path) -> List[str]:
    """
    Relocate anything an earlier run left in the deliverable directory, so
    that DATA_DIR ends up holding only the two Feather tables. Files are
    moved, never deleted.
    """
    moved = []
    old_raw = base / "raw"
    if old_raw.is_dir():
        raw_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(old_raw.iterdir()):
            if f.is_file():
                target = raw_dir / f.name
                if target.exists():
                    f.unlink()
                    moved.append(f"raw/{f.name} -> already cached, removed duplicate")
                else:
                    f.replace(target)
                    moved.append(f"raw/{f.name} -> {raw_dir}")
        try:
            old_raw.rmdir()
        except OSError:
            pass
    for name in _LEGACY_SIDECARS:
        f = base / name
        if f.is_file():
            meta_dir.mkdir(parents=True, exist_ok=True)
            target = meta_dir / name
            if target.exists():
                target.unlink()
            f.replace(target)
            moved.append(f"{name} -> {meta_dir}")
    return moved


def run(data_dir: str = DATA_DIR, raw_dir: Optional[str] = RAW_DIR,
        meta_dir: str = META_DIR, force_download: bool = False,
        timeout: int = 120, verbose: bool = False, scope: str = "conus",
        write_meta: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir).expanduser()
    meta_dir = Path(meta_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    if write_meta:
        meta_dir.mkdir(parents=True, exist_ok=True)

    # No raw_dir: download into a temp directory and remove it afterwards,
    # so the only thing this script leaves behind is the two tables.
    tmp_root = None
    if raw_dir is None:
        tmp_root = tempfile.mkdtemp(prefix="usgs_raw_")
        raw_dir = Path(tmp_root)
    else:
        raw_dir = Path(raw_dir).expanduser()
        raw_dir.mkdir(parents=True, exist_ok=True)

    relocated = ([] if tmp_root else migrate_legacy(base, raw_dir, meta_dir))

    # Only the two Feather tables are written. To bring the QA report and
    # the provenance/download CSVs back, set write_meta=True (or pass
    # --write-meta) and they land in meta_dir as before.
    rep = Report(meta_dir / "usgs_extraction_qa.txt" if write_meta else None,
                 echo=verbose)
    downloads, prov, failed = [], [], []
    try:
        rep("USGS livestock water use -- download and extraction")
        rep(f"tables dir: {base}   (Feather only)")
        rep(f"raw files:  {raw_dir}"
            + ("   (temporary, deleted after this run)" if tmp_root else ""))
        rep(f"metadata:   {meta_dir}" if write_meta
            else "metadata:   not written (write_meta=False)")
        rep(f"run (UTC): {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
        rep(f"pandas {pd.__version__} | python {sys.version.split()[0]}")
        rep("\nValues are Mgal/d. Livestock is LS/ls/LI; aquaculture is "
            "LA/la/AQ; lv/LV (their sum) is never used.")

        if relocated:
            rep("\n" + "=" * 78)
            rep("RELOCATED FROM THE TABLES DIRECTORY")
            rep("=" * 78)
            for line in relocated:
                rep(f"  {line}")

        rep("\n" + "=" * 78)
        rep("DOWNLOAD")
        rep("=" * 78)
        paths: Dict[object, Optional[Path]] = {}
        bar = tqdm(total=len(SOURCES), desc="Fetching USGS sources", unit="file")
        for key, spec in SOURCES.items():
            bar.set_postfix_str(str(key))
            path, meta = fetch(rep, key, spec, raw_dir, force_download, timeout)
            paths[key] = path
            downloads.append(meta)
            bar.update(1)
        bar.close()
        failed[:] = [d["key"] for d in downloads
                     if d["status"] == "all urls failed"]

        state_df = pd.DataFrame(columns=STATE_SCHEMA)
        if paths.get("1950-1980") is not None:
            state_df, sprov = extract_state(rep, paths["1950-1980"])
            prov += sprov
        else:
            rep("\n  STATE: source unavailable, state table not built")

        rep("\n" + "=" * 78)
        rep("COUNTY-LEVEL 1985-2015")
        rep("=" * 78)
        cframes = []
        bar = tqdm(sorted(COUNTY_SPEC), desc="County-Level 1985-2015", unit="yr")
        for year in bar:
            bar.set_postfix_str(str(year))
            path = paths.get(year)
            if path is None:
                rep(f"  {year}: source unavailable, skipped")
                continue
            df, cprov = extract_county_year(rep, path, year)
            cframes.append(df)
            prov += cprov
        bar.close()

        county_df = pd.DataFrame(columns=COUNTY_SCHEMA)
        if cframes:
            county_df = pd.concat(cframes, ignore_index=True)
            # Years with no year column contribute all-NA, which would
            # promote this to float64 on concat and store 1985.0.
            county_df["year_reported"] = pd.to_numeric(
                county_df["year_reported"], errors="coerce").astype("Int16")
            county_df = backfill_county_names(rep, county_df)
            for c in TEXT_COLUMNS:
                if c in county_df:
                    county_df[c] = county_df[c].astype("string")

        if not state_df.empty:
            for c in TEXT_COLUMNS:
                if c in state_df:
                    state_df[c] = state_df[c].astype("string")
            state_df = state_df.sort_values(["year", "state_fips"]).reset_index(drop=True)
            qa_state(rep, state_df)
            if scope == "conus":
                state_df = restrict_to_conus(rep, state_df, "state-level 1960-1980")
            state_df = add_cw_ratio(rep, state_df, "state-level 1960-1980")
            dropped = [c for c in state_df.columns if c not in STATE_OUTPUT]
            state_df = state_df[STATE_OUTPUT]
            rep(f"\n  dropped working columns: {dropped}")
            p = base / "usgs_livestock_state_1960_1980.feather"
            state_df.to_feather(p)
            rep(f"  wrote {p.name}  ({len(state_df)} rows x {state_df.shape[1]} cols)")

        if not county_df.empty:
            county_df = county_df.sort_values(["year", "fips"]).reset_index(drop=True)
            qa_county(rep, county_df)
            if scope == "conus":
                county_df = restrict_to_conus(rep, county_df, "county-level 1985-2015")
            county_df = add_cw_ratio(rep, county_df, "county-level 1985-2015")
            dropped = [c for c in county_df.columns if c not in COUNTY_OUTPUT]
            county_df = county_df[COUNTY_OUTPUT]
            rep(f"\n  dropped working columns: {dropped}")
            p = base / "usgs_livestock_county_1985_2015.feather"
            county_df.to_feather(p)
            rep(f"  wrote {p.name}  ({len(county_df)} rows x {county_df.shape[1]} cols)")

        if write_meta:
            prov_df = pd.DataFrame(prov).merge(
                pd.DataFrame(downloads)[["cache_file", "url", "sha256", "bytes",
                                         "retrieved_utc", "status"]],
                left_on="source_file", right_on="cache_file", how="left")
            prov_df.to_csv(meta_dir / "usgs_livestock_provenance.csv", index=False)
            pd.DataFrame(downloads).to_csv(meta_dir / "usgs_downloads.csv",
                                           index=False)
            rep(f"  wrote usgs_livestock_provenance.csv  ({len(prov_df)} rows)")
            rep(f"  wrote usgs_downloads.csv  ({len(downloads)} rows)")

        rep("\n" + "=" * 78)
        rep("SCHEMAS")
        rep("=" * 78)
        rep("\n  state table")
        for c in STATE_OUTPUT:
            rep(f"    {c:<30} {state_df[c].dtype if c in state_df else '-'}")
        rep("\n  county table")
        for c in COUNTY_OUTPUT:
            rep(f"    {c:<30} {county_df[c].dtype if c in county_df else '-'}")

        rep("\n" + "=" * 78)
        rep("TABLES DIRECTORY CONTENTS")
        rep("=" * 78)
        for f in sorted(base.iterdir()):
            kind = "dir " if f.is_dir() else "file"
            flag = "" if f.suffix == ".feather" and f.is_file() else "   <-- NOT A TABLE"
            rep(f"  {kind} {f.name}{flag}")
    finally:
        rep.close()
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if not verbose:
        print("Done.")
        if not state_df.empty:
            yrs = f"{int(state_df['year'].min())}-{int(state_df['year'].max())}"
            print(f"  state-level  {len(state_df):>6,} rows x {state_df.shape[1]:>2} cols  "
                  f"({yrs})  ->  usgs_livestock_state_1960_1980.feather")
        if not county_df.empty:
            yrs = f"{int(county_df['year'].min())}-{int(county_df['year'].max())}"
            print(f"  county-level {len(county_df):>6,} rows x {county_df.shape[1]:>2} cols  "
                  f"({yrs})  ->  usgs_livestock_county_1985_2015.feather")
        print(f"  scope     {scope}")
        print(f"  tables    {base}")
        if write_meta:
            print(f"  QA report {meta_dir / 'usgs_extraction_qa.txt'}")
        # Anything that went wrong still surfaces, quiet mode or not.
        if failed:
            print(f"  WARNING: sources that could not be downloaded: {failed}")
        if not county_df.empty:
            unresolved = int(county_df["county_name"].isna().sum())
            if unresolved:
                print(f"  NOTE: {unresolved:,} rows have no county name "
                      f"(retired FIPS codes; see the QA report)")

    return state_df, county_df


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download and extract USGS livestock water use.")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="deliverables: holds only the two Feather tables")
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help="keep the downloaded source files here; omitted by "
                         "default, in which case they go to a temp directory "
                         "and are deleted after the run")
    ap.add_argument("--meta-dir", default=META_DIR,
                    help="QA report, provenance, and download log")
    ap.add_argument("--force-download", action="store_true",
                    help="re-download even if a cached copy exists")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--verbose", action="store_true",
                    help="echo the full QA narrative to the console")
    ap.add_argument("--scope", choices=("conus", "all"), default="conus",
                    help="conus = 48 states + DC (default); all = keep AK, HI, "
                         "PR, VI and the territories")
    ap.add_argument("--write-meta", action="store_true",
                    help="also write the QA report and provenance CSVs")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.raw_dir, args.meta_dir,
        args.force_download, args.timeout, args.verbose, args.scope,
        args.write_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())