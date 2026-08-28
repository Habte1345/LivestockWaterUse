"""
usda_livestock.py

County-level livestock inventory from the USDA NASS Census of Agriculture,
2002-2022, for the five livestock types used by the HLWU framework.

    <DATA_DIR>/   holds ONLY the cleaned tables:
        usda_livestock_county_2002_2022.feather        all four types, long
        usda_dairy_cattle_county_2002_2022.feather     one per type, for the
        usda_beef_cattle_county_2002_2022.feather      four separate ML
        usda_poultry_county_2002_2022.feather          models
        usda_hogs_county_2002_2022.feather

    <RAW_DIR>/    cached qs.census<year>.txt.gz downloads
    <META_DIR>/   usda_extraction_qa.txt, usda_downloads.csv

--------------------------------------------------------------------------
SOURCE
--------------------------------------------------------------------------
https://www.nass.usda.gov/datasets/qs.census<year>.txt.gz for 2002, 2007,
2012, 2017 and 2022. These are the only Census of Agriculture bulk files
NASS publishes; every other file on that page is survey data and carries a
date stamp in its name.

Each file is a long fact table: one row per published statistic, ~3.2M rows
for 2002 rising to far more for 2022. There is no "population" column --
the head count is assembled by selecting the right rows.

--------------------------------------------------------------------------
SELECTION (verified against the 2002 file, taxonomy identical 2002-2022)
--------------------------------------------------------------------------
Common pins on every type:

    SECTOR_DESC          == 'ANIMALS & PRODUCTS'
    STATISTICCAT_DESC    == 'INVENTORY'      not SALES (also in HEAD)
    UNIT_DESC            == 'HEAD'           not $ and not OPERATIONS
    DOMAIN_DESC          == 'TOTAL'          not the farm-size breakdowns
    PRODN_PRACTICE_DESC  == 'ALL PRODUCTION PRACTICES'
    AGG_LEVEL_DESC       == 'COUNTY'         not STATE / ASD / NATIONAL

Then, by type:

    dairy_cattle   CATTLE   CLASS_DESC 'COWS, MILK'
    beef_cattle    CATTLE   'INCL CALVES' minus 'COWS, MILK'   (full herd)
    hogs           HOGS     'ALL CLASSES'
    poultry        CHICKENS 'LAYERS' + 'BROILERS'

Poultry combines layers and broilers into one head count because the
framework fits a single poultry WCC equation. Note that broilers are a
flow, not a stock: a house turns over five or six flocks a year, so a
point-in-time inventory understates annual throughput. Layer and broiler
counts are kept separately in the QA report so the split is recoverable.

CLASS NESTING, verified on Sioux County IA 2002 and exact to the head:

    INCL CALVES 221,653 = COWS 30,024 + (EXCL COWS) 191,629
    COWS         30,024 = COWS, MILK 20,152 + COWS, BEEF 9,872
    HOGS ALL CLASSES 869,086 = BREEDING 35,858 + MARKET 833,228

The trap: CLASS_DESC 'ALL CLASSES' under CATTLE is NOT the cattle total.
Its SHORT_DESC reads 'CATTLE, ON FEED - INVENTORY' and its
PRODN_PRACTICE_DESC is 'ON FEED'. It is feedlot cattle, already inside
(EXCL COWS). Including it would have inflated Sioux County by 149,408 head.
The 'ALL PRODUCTION PRACTICES' pin excludes it; the class list excludes it
again, deliberately.

CHICKENS classes are disjoint with no parent total, so LAYERS and BROILERS
are independent selections. PULLETS, REPLACEMENT is excluded.

--------------------------------------------------------------------------
SUPPRESSION
--------------------------------------------------------------------------
Only rows whose VALUE parses as a number are kept. Everything else -- (D)
withheld to avoid disclosing an individual operation, (Z) less than half
the rounding unit, (X), (NA), (S), blanks -- is skipped and counted in the
QA report by flag, year and type. Nothing is filled with zero: NASS
suppresses because the value is non-zero, so a zero would be a false
statement, and a county-year with no valid value is simply absent.

NASS also publishes a pseudo-county, COUNTY_CODE 998 "OTHER (COMBINED)
COUNTIES", which carries AGG_LEVEL_DESC 'COUNTY' but holds the animals from
suppressed counties pooled to the agricultural district. It is not a county
and is removed; the head count it holds is reported in the QA as the
unallocated remainder.

Usage
-----
    python usda_livestock.py
    python usda_livestock.py --scope all --verbose
    from usda_livestock import run; df = run()
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:                                    # pragma: no cover
    requests = None

try:
    from tqdm.auto import tqdm
except ImportError:                                    # pragma: no cover
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kw):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or ())

        def update(self, n=1):
            pass

        def set_postfix_str(self, s=""):
            pass

        def close(self):
            pass

        @staticmethod
        def write(msg):
            print(msg)

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/usda_census"

# RAW_DIR = None: the five qs.census*.txt.gz files are downloaded into a
# temporary directory and deleted when the run ends. Nothing is left on
# disk except the Feather tables. Pass a path (or --raw-dir) to keep them,
# which avoids re-downloading ~650 MB on every run.
RAW_DIR = None
# META_DIR is only used when write_meta=True.
META_DIR = f"{_REPO}/data/_meta/usda"

CENSUS_YEARS = (2002, 2007, 2012, 2017, 2022)
BASE_URL = "https://www.nass.usda.gov/datasets/qs.census{}.txt.gz"

# Only these columns are read. The files are large; the rest are unused.
USECOLS = [
    "SECTOR_DESC", "GROUP_DESC", "COMMODITY_DESC", "CLASS_DESC",
    "PRODN_PRACTICE_DESC", "STATISTICCAT_DESC", "UNIT_DESC", "SHORT_DESC",
    "DOMAIN_DESC", "AGG_LEVEL_DESC",
    "STATE_ANSI", "STATE_ALPHA", "STATE_NAME",
    "COUNTY_ANSI", "COUNTY_CODE", "COUNTY_NAME", "YEAR", "VALUE",
]

CHUNK_ROWS = 400_000

PINS = {
    "SECTOR_DESC": "ANIMALS & PRODUCTS",
    "STATISTICCAT_DESC": "INVENTORY",
    "UNIT_DESC": "HEAD",
    "DOMAIN_DESC": "TOTAL",
    "PRODN_PRACTICE_DESC": "ALL PRODUCTION PRACTICES",
    "AGG_LEVEL_DESC": "COUNTY",
}

# (COMMODITY_DESC, CLASS_DESC) pairs to retain. 'COWS' and 'COWS, BEEF' are
# kept as working classes only: they are the fallback route to dairy when
# 'COWS, MILK' is suppressed. They are not emitted as livestock types.
KEEP_CLASSES = {
    "CATTLE": ["INCL CALVES", "COWS, MILK", "COWS, BEEF", "COWS"],
    # BREEDING is retained as a working class, not emitted as a type: the
    # breeding share of the herd is a county attribute that drives water
    # use hard, since sows drink far more than market pigs. Treating it as
    # unknown would push that variance into the unexplainable pile.
    "HOGS": ["ALL CLASSES", "BREEDING"],
    "CHICKENS": ["LAYERS", "BROILERS"],
}

# Classes that must be present in every year, or the taxonomy has shifted
# and the run should fail rather than quietly emit less.
REQUIRED = [("CATTLE", "INCL CALVES"), ("CATTLE", "COWS, MILK"),
            ("HOGS", "ALL CLASSES"), ("CHICKENS", "LAYERS"),
            ("CHICKENS", "BROILERS")]

# One ML model is trained per type, so each also gets its own file.
LIVESTOCK_TYPES = ("dairy_cattle", "beef_cattle", "poultry", "hogs")

OUTPUT_SCHEMA = ["year", "fips", "state_abbrev", "state_name",
                 "county_name", "livestock_type", "head",
                 "layer_fraction", "breeding_fraction"]

NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# NASS district-residual pseudo-counties.
PSEUDO_COUNTY_CODES = {"998", "999"}


class Report:
    def __init__(self, path: Optional[Path] = None, echo: bool = False):
        # path=None keeps the narrative in memory, so no QA file is written.
        # Every rep() call still works; nothing needs guarding.
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


def fetch(rep: Report, year: int, raw_dir: Path, force: bool,
          timeout: int) -> Tuple[Optional[Path], dict]:
    """Reuse a cached copy unless --force-download; otherwise download."""
    dest = raw_dir / f"qs.census{year}.txt.gz"
    meta = {"year": year, "file": dest.name, "url": None, "bytes": None,
            "sha256": None, "retrieved_utc": None, "status": None}

    # A plain .txt sitting next to the .gz is honoured too, so a file
    # already unpacked by hand is not downloaded again.
    plain = raw_dir / f"qs.census{year}.txt"
    if plain.exists() and not force:
        meta.update(url="(local)", bytes=plain.stat().st_size,
                    sha256=sha256_of(plain), status="local txt")
        rep(f"  {year}: using local {plain.name} ({meta['bytes']:,} B)")
        return plain, meta

    if dest.exists() and not force:
        meta.update(url="(cached)", bytes=dest.stat().st_size,
                    sha256=sha256_of(dest), status="cached",
                    retrieved_utc=dt.datetime.fromtimestamp(
                        dest.stat().st_mtime, dt.timezone.utc
                    ).isoformat(timespec="seconds"))
        rep(f"  {year}: cached {dest.name} ({meta['bytes']:,} B, "
            f"sha256 {meta['sha256'][:12]}...)")
        return dest, meta

    if requests is None:
        rep(f"  {year}: requests not installed and no cached copy")
        meta["status"] = "no requests module"
        return None, meta

    url = BASE_URL.format(year)
    rep(f"  {year}: GET {url}")
    try:
        resp = requests.get(url, timeout=timeout, stream=True,
                            headers={"User-Agent": "LivestockWaterUse/1.0"})
        resp.raise_for_status()
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    except Exception as err:
        rep(f"       failed: {type(err).__name__}: {err}")
        meta["status"] = f"failed: {type(err).__name__}"
        return None, meta

    meta.update(url=url, bytes=dest.stat().st_size, sha256=sha256_of(dest),
                status="downloaded",
                retrieved_utc=dt.datetime.now(dt.timezone.utc)
                .isoformat(timespec="seconds"))
    rep(f"       ok: {meta['bytes']:,} B, sha256 {meta['sha256'][:12]}...")
    return dest, meta


# ---------------------------------------------------------------------------
# read and filter
# ---------------------------------------------------------------------------

def read_year(rep: Report, path: Path, year: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Stream the file in chunks, applying the taxonomy pins to each chunk so
    only the target rows are ever held in memory. The 2022 file is ~295 MB
    compressed and does not need to be materialised whole.
    """
    kept: List[pd.DataFrame] = []
    flags: Dict[str, int] = {}
    n_raw = 0

    reader = pd.read_csv(
        path, sep="\t", encoding="latin1", usecols=USECOLS, dtype=str,
        chunksize=CHUNK_ROWS, low_memory=False,
        compression="gzip" if path.suffix == ".gz" else None,
    )

    for chunk in reader:
        n_raw += len(chunk)
        for col, val in PINS.items():
            chunk = chunk[chunk[col] == val]
            if chunk.empty:
                break
        if chunk.empty:
            continue

        mask = False
        for commodity, classes in KEEP_CLASSES.items():
            mask = mask | ((chunk["COMMODITY_DESC"] == commodity)
                           & (chunk["CLASS_DESC"].isin(classes)))
        chunk = chunk[mask]
        if not chunk.empty:
            kept.append(chunk)

    if not kept:
        raise ExtractionError(
            f"{year}: no rows survived the taxonomy pins in {path.name}. "
            f"The NASS vocabulary may have changed.")

    df = pd.concat(kept, ignore_index=True)
    rep(f"  {year}: {n_raw:,} rows scanned -> {len(df):,} candidate rows")

    # Count the suppression flags BEFORE dropping them, so the QA can say
    # what was skipped rather than only how much survived.
    raw_value = df["VALUE"].astype(str).str.strip()
    numeric = pd.to_numeric(
        raw_value.str.replace(",", "", regex=False), errors="coerce")
    bad = raw_value[numeric.isna()]
    flags = {k: int(v) for k, v in bad.value_counts().head(12).items()}

    df = df.assign(VALUE=numeric)
    df = df[df["VALUE"].notna()]

    missing = [(c, k) for c, k in REQUIRED
               if not ((df["COMMODITY_DESC"] == c) & (df["CLASS_DESC"] == k)).any()]
    if missing:
        raise ExtractionError(
            f"{year}: required class(es) absent after filtering: {missing}. "
            f"Present: "
            f"{sorted(set(zip(df['COMMODITY_DESC'], df['CLASS_DESC'])))}")

    return df, flags


def build_fips(df: pd.DataFrame) -> pd.Series:
    state = df["STATE_ANSI"].astype(str).str.strip().str.zfill(2)
    county = df["COUNTY_ANSI"].astype(str).str.strip()
    county = county.str.replace(r"\.0$", "", regex=True).str.zfill(3)
    return state + county


def to_long(rep: Report, df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Pivot on class, derive beef, and return one row per county and type."""
    df = df.copy()
    df["fips"] = build_fips(df)
    df["county_code"] = (df["COUNTY_CODE"].astype(str).str.strip()
                         .str.replace(r"\.0$", "", regex=True).str.zfill(3))

    pseudo = df["county_code"].isin(PSEUDO_COUNTY_CODES)
    if pseudo.any():
        held = df.loc[pseudo].groupby(
            ["COMMODITY_DESC", "CLASS_DESC"])["VALUE"].sum()
        rep(f"  {year}: removed {int(pseudo.sum())} 'OTHER (COMBINED) "
            f"COUNTIES' rows holding pooled animals from suppressed counties")
        for (c, k), v in held.items():
            rep(f"      {c} / {k}: {v:,.0f} head unallocated")
        df = df.loc[~pseudo]

    bad_fips = ~df["fips"].str.fullmatch(r"\d{5}").fillna(False)
    if bad_fips.any():
        rep(f"  {year}: dropped {int(bad_fips.sum())} rows without a "
            f"valid 5-digit FIPS")
        df = df.loc[~bad_fips]

    ident = (df.groupby("fips")
             .agg(state_abbrev=("STATE_ALPHA", "first"),
                  state_name=("STATE_NAME", "first"),
                  county_name=("COUNTY_NAME", "first")))

    wide = (df.pivot_table(index="fips",
                           columns=["COMMODITY_DESC", "CLASS_DESC"],
                           values="VALUE", aggfunc="sum"))

    def col(commodity: str, klass: str) -> pd.Series:
        key = (commodity, klass)
        if key in wide.columns:
            return wide[key]
        return pd.Series(np.nan, index=wide.index)

    incl = col("CATTLE", "INCL CALVES")
    milk = col("CATTLE", "COWS, MILK")
    cows = col("CATTLE", "COWS")
    beefcows = col("CATTLE", "COWS, BEEF")

    # Dairy: reported directly, else recovered as COWS - COWS, BEEF.
    dairy = milk.copy()
    fallback = dairy.isna() & cows.notna() & beefcows.notna()
    dairy.loc[fallback] = (cows - beefcows).loc[fallback]

    # Beef: the full herd minus dairy cows.
    #
    # Where dairy is suppressed, beef takes the cattle total unreduced. NASS
    # suppresses 'COWS, MILK' (and, complementarily, 'COWS, BEEF') only when
    # a county has very few dairy operations -- median 2, 75th percentile 4
    # across the 562 affected counties in 2002. Those counties hold 14.25M
    # of the 95.3M national herd, so discarding them would remove 15% of the
    # cattle. The alternative lower bound, total minus all cows, is useless
    # here because cows are 54% of the total in exactly these cow-calf
    # counties. Taking the total instead carries a small, one-directional
    # bias: a few small dairy herds counted as beef.
    beef = incl - dairy.fillna(0)
    beef.loc[incl.isna()] = np.nan
    absorbed = incl.notna() & dairy.isna()

    rep(f"  {year}: dairy reported {int(milk.notna().sum()):,} | "
        f"recovered via COWS-COWS,BEEF {int(fallback.sum()):,} | "
        f"dairy suppressed, beef takes the full total {int(absorbed.sum()):,}")

    neg = beef < 0
    if neg.any():
        rep(f"  {year}: {int(neg.sum())} counties where dairy exceeds the "
            f"cattle total (source inconsistency); beef set to missing")
        beef.loc[neg] = np.nan

    layers = col("CHICKENS", "LAYERS")
    broilers = col("CHICKENS", "BROILERS")
    # Poultry is layers + broilers. A county reporting only one of the two
    # keeps that one rather than becoming missing, so min_count=1 rather
    # than a plain sum (which would turn NaN + NaN into 0).
    poultry = pd.concat([layers, broilers], axis=1).sum(axis=1, min_count=1)

    rep(f"  {year}: poultry = layers ({int(layers.notna().sum()):,} counties) "
        f"+ broilers ({int(broilers.notna().sum()):,}) -> "
        f"{int(poultry.notna().sum()):,} counties; "
        f"both present in {int((layers.notna() & broilers.notna()).sum()):,}")

    hogs_all = col("HOGS", "ALL CLASSES")
    hogs_breeding = col("HOGS", "BREEDING")

    out = pd.DataFrame({
        "dairy_cattle": dairy,
        "beef_cattle": beef,
        "hogs": hogs_all,
        "poultry": poultry,
        # kept for the QA split only, dropped before the table is written
        "_layers": layers,
        "_broilers": broilers,
    })

    # Herd composition: observable county attributes, carried alongside the
    # head counts because they drive per-head water use.
    composition = pd.DataFrame({
        "layer_fraction": layers / poultry.where(poultry > 0),
        "breeding_fraction": hogs_breeding / hogs_all.where(hogs_all > 0),
    })
    rep(f"  {year}: layer fraction reported for "
        f"{int(composition['layer_fraction'].notna().sum()):,} counties, "
        f"breeding fraction for "
        f"{int(composition['breeding_fraction'].notna().sum()):,}")

    # melt rather than stack: stack's dropna argument behaves differently
    # across pandas versions and was removed in 3.0.
    long = (out.reset_index()
            .melt(id_vars="fips", var_name="livestock_type", value_name="head")
            .dropna(subset=["head"]))
    long = long.merge(ident, left_on="fips", right_index=True, how="left")
    long = long.merge(composition, left_on="fips", right_index=True, how="left")

    # Each fraction belongs to one type only; blank it elsewhere so the
    # column cannot be misread as applying to cattle.
    long.loc[long["livestock_type"] != "poultry", "layer_fraction"] = np.nan
    long.loc[long["livestock_type"] != "hogs", "breeding_fraction"] = np.nan
    long["year"] = np.int16(year)
    return long[OUTPUT_SCHEMA]


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def qa_poultry_split(rep: Report, df: pd.DataFrame) -> None:
    """Layers and broilers separately, so the combination is auditable."""
    rep("\n" + "=" * 78)
    rep("POULTRY SPLIT (reported here only; the table carries the combined count)")
    rep("=" * 78)
    sub = df[df["livestock_type"].isin(["_layers", "_broilers", "poultry"])]
    if sub.empty:
        rep("  no poultry rows")
        return
    rep("\n  counties reporting")
    rep(sub.pivot_table(index="livestock_type", columns="year",
                        values="fips", aggfunc="nunique").to_string())
    rep("\n  national head count")
    tot = sub.pivot_table(index="livestock_type", columns="year",
                          values="head", aggfunc="sum")
    rep(tot.apply(lambda s: s.map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-"))
        .to_string())
    rep("\n  Broilers are a point-in-time inventory, not annual production: a")
    rep("  broiler house turns over five or six flocks a year, so this count")
    rep("  is the standing population, which is the right basis for a daily")
    rep("  water coefficient but not for annual throughput.")


def qa(rep: Report, df: pd.DataFrame, flags: Dict[int, Dict[str, int]]) -> None:
    rep("\n" + "=" * 78)
    rep("QA")
    rep("=" * 78)

    rep("\n  counties reporting, by type and year")
    piv = df.pivot_table(index="livestock_type", columns="year",
                         values="fips", aggfunc="nunique")
    rep(piv.to_string())

    rep("\n  national head count, by type and year")
    tot = df.pivot_table(index="livestock_type", columns="year",
                         values="head", aggfunc="sum")
    rep(tot.apply(lambda s: s.map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-"))
        .to_string())

    rep("\n  suppressed and non-numeric VALUE flags skipped, by year")
    for year in sorted(flags):
        if flags[year]:
            rep(f"  {year}: {flags[year]}")
        else:
            rep(f"  {year}: none")
    rep("  (D) is withheld to avoid disclosing an individual operation; "
        "(Z) is a value")
    rep("  below half the rounding unit. Neither is zero, and neither is "
        "filled in.")

    rep("\n  head-count distribution by type (all years pooled)")
    rep(f"  {'type':<14}{'n':>8}{'min':>10}{'median':>12}{'max':>14}")
    for t, g in df.groupby("livestock_type"):
        rep(f"  {t:<14}{len(g):>8}{g['head'].min():>10,.0f}"
            f"{g['head'].median():>12,.0f}{g['head'].max():>14,.0f}")

    rep("\n  counties present in all five census years, by type")
    for t, g in df.groupby("livestock_type"):
        per_year = g.groupby("year")["fips"].apply(set)
        if len(per_year) == len(CENSUS_YEARS):
            common = set.intersection(*per_year.tolist())
            allf = set.union(*per_year.tolist())
            rep(f"  {t:<14} {len(common):>5} of {len(allf):>5} distinct "
                f"counties appear in all 5 censuses")
        else:
            rep(f"  {t:<14} present in only {len(per_year)} census years")

    rep("\n  herd composition (observable county attributes, not sampled)")
    for col, t in (("layer_fraction", "poultry"),
                   ("breeding_fraction", "hogs")):
        if col in df:
            v = df.loc[df["livestock_type"] == t, col].dropna()
            if len(v):
                rep(f"    {col:<18} n={len(v):,}  min {v.min():.3f}  "
                    f"median {v.median():.3f}  max {v.max():.3f}")
            else:
                rep(f"    {col:<18} not reported")

    rep("\n  zero head counts (a reported zero, not a suppression)")
    z = df[df["head"] == 0].groupby("livestock_type").size()
    rep(f"    {dict(z) if len(z) else 'none'}")


def restrict_to_conus(rep: Report, df: pd.DataFrame) -> pd.DataFrame:
    keep = ~df["fips"].str[:2].isin(NON_CONUS_FIPS)
    removed = df.loc[~keep]
    rep("\n" + "=" * 78)
    rep("CONUS RESTRICTION")
    rep("=" * 78)
    rep(f"  kept {int(keep.sum()):,} rows, removed {int((~keep).sum()):,}")
    if not removed.empty:
        rep(f"  removed by state: "
            f"{dict(removed.groupby('state_abbrev').size().sort_values(ascending=False))}")
    rep(f"  distinct states retained: {df.loc[keep, 'fips'].str[:2].nunique()} "
        f"(expected 49: 48 states + DC)")
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(data_dir: str = DATA_DIR, raw_dir: Optional[str] = RAW_DIR,
        meta_dir: str = META_DIR, force_download: bool = False,
        timeout: int = 600, verbose: bool = False,
        scope: str = "conus", write_meta: bool = False) -> pd.DataFrame:
    base = Path(data_dir).expanduser()
    meta = Path(meta_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    if write_meta:
        meta.mkdir(parents=True, exist_ok=True)

    # No raw_dir: download into a temp directory and remove it afterwards.
    tmp_root = None
    if raw_dir is None:
        tmp_root = tempfile.mkdtemp(prefix="usda_raw_")
        raw = Path(tmp_root)
    else:
        raw = Path(raw_dir).expanduser()
        raw.mkdir(parents=True, exist_ok=True)

    # Only the Feather tables are written by default. write_meta=True (or
    # --write-meta) restores the QA report and the download log.
    rep = Report(meta / "usda_extraction_qa.txt" if write_meta else None,
                 echo=verbose)
    downloads, flags = [], {}
    df = pd.DataFrame(columns=OUTPUT_SCHEMA)
    try:
        rep("USDA NASS Census of Agriculture -- livestock inventory")
        rep(f"tables dir: {base}   (Feather only)")
        rep(f"raw files:  {raw}"
            + ("   (temporary, deleted after this run)" if tmp_root else ""))
        rep(f"metadata:   {meta}" if write_meta
            else "metadata:   not written (write_meta=False)")
        rep(f"run (UTC):  {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
        rep(f"pandas {pd.__version__} | python {sys.version.split()[0]}")
        rep(f"spatial scope: {scope}")
        rep("\nUnits are head. Beef cattle is the full herd minus dairy cows: "
            "CATTLE 'INCL CALVES' minus 'COWS, MILK'.")

        rep("\n" + "=" * 78)
        rep("DOWNLOAD")
        rep("=" * 78)
        paths: Dict[int, Optional[Path]] = {}
        bar = tqdm(total=len(CENSUS_YEARS), desc="Fetching NASS census", unit="file")
        for year in CENSUS_YEARS:
            bar.set_postfix_str(str(year))
            p, m = fetch(rep, year, raw, force_download, timeout)
            paths[year] = p
            downloads.append(m)
            bar.update(1)
        bar.close()

        rep("\n" + "=" * 78)
        rep("EXTRACTION")
        rep("=" * 78)
        frames = []
        bar = tqdm(CENSUS_YEARS, desc="Parsing census years", unit="yr")
        for year in bar:
            bar.set_postfix_str(str(year))
            if paths[year] is None:
                rep(f"  {year}: source unavailable, skipped")
                continue
            raw_df, yr_flags = read_year(rep, paths[year], year)
            flags[year] = yr_flags
            frames.append(to_long(rep, raw_df, year))
        bar.close()

        if not frames:
            rep("\nNo years extracted.")
            return df

        df = pd.concat(frames, ignore_index=True)
        if scope == "conus":
            df = restrict_to_conus(rep, df)

        df["head"] = df["head"].round().astype("Int64")
        for c in ("fips", "state_abbrev", "state_name", "county_name",
                  "livestock_type"):
            df[c] = df[c].astype("string")
        df = (df.sort_values(["year", "livestock_type", "fips"])
              .reset_index(drop=True))[OUTPUT_SCHEMA]

        qa_poultry_split(rep, df)
        df = df[~df["livestock_type"].str.startswith("_")].reset_index(drop=True)

        qa(rep, df, flags)

        rep("\n" + "=" * 78)
        rep("OUTPUT FILES")
        rep("=" * 78)
        out = base / "usda_livestock_county_2002_2022.feather"
        df.to_feather(out)
        rep(f"  {out.name}  ({len(df):,} rows x {df.shape[1]} cols, all types)")

        # One file per livestock type: four separate ML models are trained,
        # one per type, so each gets its own table.
        for t in LIVESTOCK_TYPES:
            sub = (df[df["livestock_type"] == t]
                   .drop(columns=["livestock_type"])
                   .reset_index(drop=True))
            # Drop composition columns that do not apply to this type.
            sub = sub.drop(columns=[c for c in
                                    ("layer_fraction", "breeding_fraction")
                                    if c in sub and sub[c].isna().all()])
            if sub.empty:
                rep(f"  {t}: no rows, file not written")
                continue
            p_out = base / f"usda_{t}_county_2002_2022.feather"
            sub.to_feather(p_out)
            rep(f"  {p_out.name}  ({len(sub):,} rows x {sub.shape[1]} cols, "
                f"{sub['fips'].nunique():,} counties, "
                f"{int(sub['head'].sum()):,} head in {len(CENSUS_YEARS)} censuses)")
        if write_meta:
            pd.DataFrame(downloads).to_csv(meta / "usda_downloads.csv",
                                           index=False)
            rep(f"  wrote usda_downloads.csv  ({len(downloads)} rows)")

        rep("\n" + "=" * 78)
        rep("SCHEMA")
        rep("=" * 78)
        for c in OUTPUT_SCHEMA:
            rep(f"    {c:<18} {df[c].dtype}")

        rep("\n" + "=" * 78)
        rep("TABLES DIRECTORY CONTENTS")
        rep("=" * 78)
        for f in sorted(base.iterdir()):
            flag = "" if f.suffix == ".feather" else "   <-- NOT A TABLE"
            rep(f"  {f.name}{flag}")
    finally:
        rep.close()
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if not verbose:
        print("Done.")
        if not df.empty:
            print(f"  combined   {len(df):>8,} rows x {df.shape[1]} cols  "
                  f"({df['year'].min()}-{df['year'].max()}, "
                  f"{df['livestock_type'].nunique()} types, "
                  f"{df['fips'].nunique():,} counties)")
            for t in LIVESTOCK_TYPES:
                sub = df[df["livestock_type"] == t]
                if not sub.empty:
                    print(f"  {t:<11}{len(sub):>8,} rows"
                          f"  ({sub['fips'].nunique():,} counties)"
                          f"  ->  usda_{t}_county_2002_2022.feather")
        print(f"  scope      {scope}")
        print(f"  tables     {base}")
        if write_meta:
            print(f"  QA report  {meta / 'usda_extraction_qa.txt'}")
        failed = [d["year"] for d in downloads
                  if d["status"] and d["status"].startswith("failed")]
        if failed:
            print(f"  WARNING: years that could not be downloaded: {failed}")

    return df


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract USDA NASS county livestock inventory 2002-2022.")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help="keep the downloaded .txt.gz files here; omitted by "
                         "default, in which case they go to a temp directory "
                         "and are deleted after the run")
    ap.add_argument("--meta-dir", default=META_DIR)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--scope", choices=("conus", "all"), default="conus")
    ap.add_argument("--write-meta", action="store_true",
                    help="also write the QA report and the download log")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.raw_dir, args.meta_dir, args.force_download,
        args.timeout, args.verbose, args.scope, args.write_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())