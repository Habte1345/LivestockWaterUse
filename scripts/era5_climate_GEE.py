"""
era5_forcings.py

Monthly ERA5-Land forcings for CONUS counties and states, plus every derived
variable the HLWU water-consumption surrogates actually consume.

One cleaned Feather per forcing, covering every year, at county level with
the matching state value carried alongside in the same row:

    <DATA_DIR>/
        Precp_county_level_1960_2022.feather
        Temp_county_level_1960_2022.feather
        RH_county_level_1960_2022.feather
        Wind_county_level_1960_2022.feather
        Sunlight_county_level_1960_2022.feather
        Radiation_county_level_1960_2022.feather
        CETI_county_level_1960_2022.feather
        Area_county_level_1960_2022.feather      static geometry

Every file has the same key columns -- year, month, fips, state_fips,
county_name, state_name -- so they join to each other and to the USGS and
USDA tables on fips. Nothing else is written: no raw downloads, no
per-year parts, no cache directory.

--------------------------------------------------------------------------
WHY MONTHLY AND NOT ANNUAL
--------------------------------------------------------------------------
The water-to-feed ratio is CONVEX in temperature: it is roughly flat to
20 degC then accelerates, reaching 4.8 for swine and 8.0 for layers under
heat load. For a convex function, mean(f(T)) > f(mean(T)). Feeding annual
mean temperature into the WCC surrogate therefore UNDERSTATES water use,
and it understates it most where the seasonal range is widest -- which is
also where the summer peak that drives livestock water demand actually
happens.

The correct order is: extract monthly forcings, evaluate the WCC monthly,
then aggregate. This module produces the monthly table so that the
downstream step can do exactly that. Annual tables are also written, for
context and for the ratio predictors, but they are not the right input to
a convex coefficient.

This is the same "mean of a function is not the function of the mean"
problem that applies to relative humidity, and for the same reason RH is
computed per month here rather than from annual means.

--------------------------------------------------------------------------
VARIABLES, AND WHY EACH ONE IS HERE
--------------------------------------------------------------------------
temp_c         2 m air temperature. Direct WCC predictor for every type.
rh_pct         Relative humidity, Magnus-Tetens from T and Td, computed per
               MONTH. Enters CETI and therefore the beef surrogate.
wind10_ms      10 m wind speed from the u and v components.
wind2_kmh      Wind at 2 m via the FAO-56 log profile, u2 = u10 * 0.748.
               NASEM CETI is built on near-animal wind, not 10 m wind.
rs_mj_m2_day   Downwelling shortwave at the surface.
sunlight_h     Actual sunshine hours from the Angstrom-Prescott relation,
               n = N * (Rs/Ra - 0.25) / 0.50, with extraterrestrial
               radiation Ra and daylength N computed astronomically from
               the county centroid latitude (FAO-56 Ch.3). CETI needs
               hours of sunlight, and an annual mean daylength is close to
               12 h at every latitude, so daylength alone carries almost no
               spatial signal -- actual sunshine duration does.
ceti           NASEM (2016) Eq 19-103, evaluated per month from the four
               variables above. This is what the beef water intake
               equation consumes; without wind it could not be computed at
               all, which is why the previous extraction could not feed the
               beef surrogate.
precip_mm      Monthly total. Not a WCC predictor, carried as context and
               for the precipitation ratio.

--------------------------------------------------------------------------
Everything corrected in the previous version is retained: FIPS join keys,
CONUS filtering before reduction, per-month RH, Kelvin temperature ratios,
a fine-scale retry for counties too small to contain a pixel centre,
per-year caching with backoff retries, static areas, and an explicit
Earth Engine project.

Boundary caveat: TIGER 2018 geometry is applied to every year. County
boundaries changed over the period, so early years are approximations
against modern boundaries -- the same caveat as the USGS county series.

Usage
-----
    python era5_forcings.py --project <ee-project>
    python era5_forcings.py --project <ee-project> --start 1985
    from era5_forcings import run; run(project="...")
"""

from __future__ import annotations

import argparse
import calendar
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:                                     # pragma: no cover
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

_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/climate"

# One Feather per forcing, all years, county rows carrying the matching
# state values. Nothing else is written to disk: no raw downloads, no
# per-year parts, no cache. Years are accumulated in memory and written
# once at the end.
#
# The trade-off is deliberate but worth stating: without a per-year cache
# an interruption late in the run loses the whole run. Requests are
# retried with backoff to make that unlikely, and --start/--end let a
# shorter span be pulled if it becomes a problem.
FILE_STEM = "{var}_county_level_{start}_{end}.feather"

COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"

# Source bands. Validated against the live collection at startup so a
# renamed band fails immediately instead of producing empty columns.
SRC_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
]
SHORT = {
    "temperature_2m": "t2m",
    "dewpoint_temperature_2m": "d2m",
    "total_precipitation_sum": "tp",
    "u_component_of_wind_10m": "u10",
    "v_component_of_wind_10m": "v10",
    "surface_solar_radiation_downwards_sum": "ssrd",
}

NATIVE_SCALE = 11132        # ERA5-Land grid spacing
FALLBACK_SCALE = 500        # fine retry for counties with no pixel centre

NON_CONUS_FIPS = ["02", "15", "60", "66", "69", "72", "78"]
START_YEAR, END_YEAR = 1960, 2022
KELVIN = 273.15

# FAO-56 Ch.3 constants
GSC = 0.0820               # solar constant, MJ m-2 min-1
ANGSTROM_A, ANGSTROM_B = 0.25, 0.50
WIND_10M_TO_2M = 0.748     # FAO-56 Eq 47 at z = 10 m


class Report:
    def __init__(self, echo: bool = False):
        self.echo = echo

    def __call__(self, text: str = "") -> None:
        if self.echo:
            tqdm.write(text)


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------

def init_ee(project: Optional[str]):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def validate_bands(ee, rep: Report) -> None:
    """
    Fail loudly if the collection no longer carries a band we rely on.
    A silently missing band would otherwise produce a column of nulls and
    a CETI that cannot be computed.
    """
    have = set(ee.ImageCollection(COLLECTION).first().bandNames().getInfo())
    missing = [b for b in SRC_BANDS if b not in have]
    if missing:
        raise RuntimeError(
            f"{COLLECTION} is missing expected band(s): {missing}\n"
            f"Available: {sorted(have)}")
    rep(f"  all {len(SRC_BANDS)} source bands present in {COLLECTION}")


def year_stack(ee, year: int):
    """
    One image per year whose bands are <short>_<month>, so a whole year of
    monthly values comes back in a single reduceRegions call. Twelve
    separate calls per year would be 756 round trips instead of 63.
    """
    coll = ee.ImageCollection(COLLECTION).select(SRC_BANDS)
    bands = []
    for m in range(1, 13):
        start = ee.Date.fromYMD(year, m, 1)
        img = coll.filterDate(start, start.advance(1, "month")).first()
        renamed = img.select(SRC_BANDS,
                             [f"{SHORT[b]}_{m:02d}" for b in SRC_BANDS])
        bands.append(renamed)
    return ee.Image.cat(bands).set("year", year)


def build_regions(ee):
    """CONUS counties and states with FIPS keys, static area and centroid."""
    not_conus = ee.Filter.inList("STATEFP", NON_CONUS_FIPS).Not()

    def enrich(f):
        c = f.geometry().centroid(ee.ErrorMargin(10)).coordinates()
        return f.set({
            "area_m2": f.area(ee.ErrorMargin(10)),
            "lon": ee.List(c).get(0),
            "lat": ee.List(c).get(1),
        })

    states = (ee.FeatureCollection("TIGER/2018/States")
              .filter(not_conus)
              .map(lambda f: enrich(f).set({"fips": f.get("STATEFP")})))

    counties = (ee.FeatureCollection("TIGER/2018/Counties")
                .filter(not_conus)
                .map(lambda f: enrich(f).set({
                    "fips": f.get("GEOID"),
                    "state_fips": f.get("STATEFP"),
                })))
    return counties, states


def value_columns() -> List[str]:
    return [f"{SHORT[b]}_{m:02d}" for m in range(1, 13) for b in SRC_BANDS]


def reduce_year(ee, img, regions, keep: List[str], scale: int) -> pd.DataFrame:
    fc = img.reduceRegions(collection=regions, reducer=ee.Reducer.mean(),
                           scale=scale, tileScale=4)
    props = keep + value_columns()
    rows = fc.select(props, retainGeometry=False).getInfo()["features"]
    return pd.DataFrame([r["properties"] for r in rows])


def with_retry(fn, attempts: int = 5, base_delay: float = 4.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as err:                # noqa: BLE001 - EE raises broadly
            last = err
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"Earth Engine call failed after {attempts} tries: {last}")


def fetch_level(ee, regions, keep: List[str], level: str, years: range,
                rep: Report) -> pd.DataFrame:
    """
    Pull every year and accumulate in memory. Nothing is written to disk
    here: the only artefacts of this module are the cleaned per-forcing
    tables at the end.
    """
    frames = []
    bar = tqdm(list(years), desc=f"ERA5 {level}", unit="yr")
    for year in bar:
        bar.set_postfix_str(str(year))
        img = year_stack(ee, year)
        df = with_retry(lambda: reduce_year(ee, img, regions, keep, NATIVE_SCALE))

        probe = f"{SHORT['temperature_2m']}_07"
        if probe in df:
            missing = df[probe].isna()
            if missing.any():
                ids = df.loc[missing, "fips"].tolist()
                rep(f"  {level} {year}: {len(ids)} region(s) held no pixel "
                    f"centre at {NATIVE_SCALE} m; retrying at {FALLBACK_SCALE} m")
                small = regions.filter(ee.Filter.inList("fips", ids))
                fix = with_retry(lambda: reduce_year(
                    ee, img, small, keep, FALLBACK_SCALE))
                if not fix.empty:
                    df = pd.concat([df.loc[~missing], fix], ignore_index=True)

        df["year"] = np.int16(year)
        frames.append(df)
    bar.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_long(wide: pd.DataFrame, keep: List[str]) -> pd.DataFrame:
    """Turn the <short>_<month> stack into one row per region-month."""
    out = []
    for m in range(1, 13):
        cols = {f"{SHORT[b]}_{m:02d}": SHORT[b] for b in SRC_BANDS}
        present = [c for c in cols if c in wide.columns]
        sub = wide[keep + present].rename(columns=cols).copy()
        sub["month"] = np.int8(m)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def magnus_ratio_rh(t_c, td_c):
    """
    RH = 100 * e(Td)/e(T) with Magnus-Tetens. The 6.1078 coefficient
    cancels, so only the exponent difference is evaluated.
    """
    a_td = 7.5 * td_c / (td_c + 237.3)
    a_t = 7.5 * t_c / (t_c + 237.3)
    return np.clip(100.0 * np.power(10.0, a_td - a_t), 0.0, 100.0)


def extraterrestrial_radiation(lat_deg, doy):
    """FAO-56 Eq 21: Ra in MJ m-2 day-1, and Eq 34: daylength N in hours."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    j = np.asarray(doy, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    dec = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0)
    ws = np.arccos(x)
    ra = (24.0 * 60.0 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(dec)
        + np.cos(phi) * np.cos(dec) * np.sin(ws))
    n_hours = 24.0 / np.pi * ws
    return ra, n_hours


def ceti(tc, rh, ws_kmh, hrs):
    """
    NASEM (2016) Eq 19-103, Current Effective Temperature Index. Verified
    against the worked example in NMSU Guide B-231: Tc 26.7, RH 30,
    WS 10 km/h, HRS 12 gives 29.10 (published 29.1).
    """
    tc = np.asarray(tc, dtype=float)
    rh = np.asarray(rh, dtype=float)
    ws = np.asarray(ws_kmh, dtype=float) / 3.6          # km/h -> m/s
    hrs = np.asarray(hrs, dtype=float)
    return (27.88 - 0.456 * tc + 0.010754 * tc ** 2
            - 0.4905 * rh + 0.00088 * rh ** 2
            + 1.1507 * ws - 0.126447 * ws ** 2
            + 0.019867 * tc * rh - 0.046313 * tc * ws
            + 0.41267 * hrs)


# Refuse to run if the published worked example does not reproduce.
_chk = float(ceti(26.7, 30.0, 10.0, 12.0))
assert abs(_chk - 29.1) < 0.05, f"CETI check failed: {_chk}"


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Units and every derived forcing, evaluated per region-month."""
    out = df.copy()
    out["temp_c"] = out["t2m"] - KELVIN
    out["dewpoint_c"] = out["d2m"] - KELVIN
    out["rh_pct"] = magnus_ratio_rh(out["temp_c"], out["dewpoint_c"])

    # total_precipitation_sum is a FLOW band, so MONTHLY_AGGR carries the
    # monthly total in metres, not a rate.
    out["precip_mm"] = out["tp"] * 1000.0

    out["wind10_ms"] = np.hypot(out["u10"], out["v10"])
    out["wind2_kmh"] = out["wind10_ms"] * WIND_10M_TO_2M * 3.6

    # surface_solar_radiation_downwards_sum is also a flow band: J m-2
    # accumulated over the month. Convert to a mean daily MJ m-2 day-1.
    days = out["month"].map(lambda m: calendar.monthrange(2001, int(m))[1])
    out["rs_mj_m2_day"] = out["ssrd"] / days / 1e6

    # Angstrom-Prescott inverted for actual sunshine hours.
    mid_doy = out["month"].map(
        lambda m: int(pd.Timestamp(2001, int(m), 15).dayofyear))
    ra, n_hours = extraterrestrial_radiation(out["lat"], mid_doy)
    out["ra_mj_m2_day"] = ra
    out["daylength_h"] = n_hours
    frac = np.clip(
        (out["rs_mj_m2_day"] / np.where(ra > 0, ra, np.nan) - ANGSTROM_A)
        / ANGSTROM_B, 0.0, 1.0)
    out["sunlight_h"] = frac * n_hours

    out["ceti"] = ceti(out["temp_c"], out["rh_pct"],
                       out["wind2_kmh"], out["sunlight_h"])
    out["area_km2"] = out["area_m2"] / 1e6
    return out


# One output file per forcing. `col` is the derived column; `how`
# controls annual aggregation; `pair` says which county/state comparison
# is meaningful for that variable.
#
# Temperature and CETI use a DIFFERENCE, not a ratio: both are interval
# scales whose zero point is arbitrary, so a ratio of them is not a
# physical quantity and diverges when the state value nears zero.
# Precipitation, humidity, wind and radiation have a true zero, so their
# ratios are meaningful.
FORCINGS = [
    ("Precp",     "precip_mm",     "sum",  "ratio"),
    ("Temp",      "temp_c",        "mean", "diff"),
    ("RH",        "rh_pct",        "mean", "ratio"),
    ("Wind",      "wind2_kmh",     "mean", "ratio"),
    ("Sunlight",  "sunlight_h",    "mean", "ratio"),
    ("Radiation", "rs_mj_m2_day",  "mean", "ratio"),
    ("CETI",      "ceti",          "mean", "diff"),
]

KEYS = ["year", "month", "fips", "state_fips", "county_name", "state_name"]


def annualise(monthly: pd.DataFrame, keys: List[str],
              col: str, how: str) -> pd.DataFrame:
    """Precipitation sums over the year; everything else is a monthly mean."""
    out = monthly.groupby(keys, as_index=False)[col].agg(how)
    peak = (monthly.groupby(keys, as_index=False)[col]
            .max().rename(columns={col: f"{col}_max_month"}))
    return out.merge(peak, on=keys, how="left")


def build_forcing_table(cm: pd.DataFrame, sm: pd.DataFrame, col: str,
                        how: str, pair: str, monthly: bool) -> pd.DataFrame:
    """
    One tidy table for a single forcing: the county value, the value for
    the state it sits in, and their comparison, all on the same row.
    """
    ckeys = ["year", "fips", "state_fips", "county_name"]
    skeys = ["year", "fips", "state_name"]
    if monthly:
        ckeys.insert(1, "month")
        skeys.insert(1, "month")
        c = cm[ckeys + [col]].copy()
        st = sm[skeys + [col]].copy()
    else:
        c = annualise(cm, ckeys, col, how)
        st = annualise(sm, skeys, col, how)

    st = st.rename(columns={"fips": "state_fips"})
    join = ["year", "state_fips"] + (["month"] if monthly else [])
    m = c.merge(st, on=join, how="left", suffixes=("_county", "_state"))

    if pair == "ratio":
        denom = m[f"{col}_state"].where(m[f"{col}_state"] != 0)
        m[f"{col}_ratio"] = m[f"{col}_county"] / denom
    else:
        m[f"{col}_diff"] = m[f"{col}_county"] - m[f"{col}_state"]

    order = (["year"] + (["month"] if monthly else [])
             + ["fips", "state_fips", "county_name", "state_name"])
    rest = [c_ for c_ in m.columns if c_ not in order]
    return m[order + rest].sort_values(order[:3]).reset_index(drop=True)


def build_area_table(cm: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    """Static geometry: it does not vary by year, so it gets its own file."""
    c = (cm[["fips", "state_fips", "county_name", "lat", "lon", "area_km2"]]
         .drop_duplicates("fips"))
    st = (sm[["fips", "state_name", "area_km2"]]
          .drop_duplicates("fips").rename(columns={"fips": "state_fips"}))
    m = c.merge(st, on="state_fips", how="left", suffixes=("_county", "_state"))
    m["area_ratio"] = m["area_km2_county"] / m["area_km2_state"]
    return m.sort_values("fips").reset_index(drop=True)


def qa(rep: Report, cm: pd.DataFrame, sm: pd.DataFrame,
       tables: Dict[str, pd.DataFrame]) -> None:
    rep("\n" + "=" * 78)
    rep("QA")
    rep("=" * 78)
    rep(f"  counties {cm['fips'].nunique():,} | states {sm['fips'].nunique()} "
        f"(expected 49: 48 states + DC)")
    rep(f"  county rows {len(cm):,} | state rows {len(sm):,}")

    rep("\n  monthly county ranges (plausible bounds in brackets)")
    bounds = {
        "temp_c": (-45, 45, "degC"), "rh_pct": (5, 100, "%"),
        "precip_mm": (0, 1200, "mm/month"), "wind2_kmh": (0, 40, "km/h"),
        "rs_mj_m2_day": (1, 35, "MJ/m2/d"), "sunlight_h": (0, 16, "h"),
        "ceti": (-10, 45, "index"),
    }
    for col, (lo, hi, unit) in bounds.items():
        if col not in cm:
            continue
        v = cm[col]
        bad = int(((v < lo) | (v > hi)).sum())
        rep(f"    {col:<14} min {v.min():>8.2f}  median {v.median():>8.2f}"
            f"  max {v.max():>8.2f} {unit:<10} outside [{lo},{hi}]: {bad:,}"
            f"  nulls {int(v.isna().sum()):,}")

    rep("\n  why the tables are monthly: the water-to-feed response is convex")
    if "temp_c" in cm:
        wfr = lambda t: np.interp(t, [0, 15, 20, 25, 30, 35, 40],
                                  [2.0, 2.1, 2.4, 2.9, 3.5, 4.2, 4.8])
        g = cm.groupby(["fips", "year"])["temp_c"]
        gap = g.apply(lambda x: np.mean(wfr(x))) - g.mean().map(wfr)
        rep(f"    mean(WFR(T_month)) - WFR(mean T): median {gap.median():+.4f}, "
            f"p95 {gap.quantile(.95):+.4f}")
        rep("    positive means an annual mean would UNDERSTATE water use.")

    rep("\n  output tables")
    for name, df in tables.items():
        comp = [c for c in df.columns if c.endswith(("_ratio", "_diff"))]
        line = f"    {name:<44} {len(df):>9,} rows x {df.shape[1]:>2} cols"
        if comp:
            v = df[comp[0]].dropna()
            if len(v):
                line += (f"   {comp[0]}: median {v.median():.4f}"
                         f"  [{v.min():.3f}, {v.max():.3f}]")
        rep(line)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(project: Optional[str] = None, data_dir: str = DATA_DIR,
        start: int = START_YEAR, end: int = END_YEAR,
        freq: str = "monthly", verbose: bool = False) -> Dict[str, pd.DataFrame]:
    base = Path(data_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    rep = Report(echo=verbose)
    monthly = freq == "monthly"

    ee = init_ee(project)
    validate_bands(ee, rep)
    counties, states = build_regions(ee)
    years = range(start, end + 1)

    c_keep = ["fips", "state_fips", "NAME", "area_m2", "lat", "lon"]
    s_keep = ["fips", "NAME", "area_m2", "lat", "lon"]
    c_wide = fetch_level(ee, counties, c_keep, "county", years, rep)
    s_wide = fetch_level(ee, states, s_keep, "state", years, rep)

    cm = derive(to_long(c_wide, [c for c in c_keep] + ["year"]))
    sm = derive(to_long(s_wide, [c for c in s_keep] + ["year"]))
    cm = cm.rename(columns={"NAME": "county_name"})
    sm = sm.rename(columns={"NAME": "state_name"})

    cm["fips"] = cm["fips"].astype(str).str.zfill(5)
    cm["state_fips"] = cm["state_fips"].astype(str).str.zfill(2)
    sm["fips"] = sm["fips"].astype(str).str.zfill(2)

    tables: Dict[str, pd.DataFrame] = {}
    for label, col, how, pair in FORCINGS:
        if col not in cm:
            continue
        name = FILE_STEM.format(var=label, start=start, end=end)
        tables[name] = build_forcing_table(cm, sm, col, how, pair, monthly)

    area_name = FILE_STEM.format(var="Area", start=start, end=end)
    tables[area_name] = build_area_table(cm, sm)

    qa(rep, cm, sm, tables)

    for name, df in tables.items():
        df.reset_index(drop=True).to_feather(base / name)

    if not verbose:
        print("Done.")
        for name, df in tables.items():
            print(f"  {name:<44} {len(df):>9,} rows x {df.shape[1]:>2} cols")
        print(f"  granularity  {'county-month' if monthly else 'county-year'}")
        print(f"  counties {cm['fips'].nunique():,} | states "
              f"{sm['fips'].nunique()} | {start}-{end}")
        print(f"  tables   {base}")
        n = int(cm["temp_c"].isna().sum())
        if n:
            print(f"  WARNING: {n:,} county-months with no value")
    return tables


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="ERA5-Land monthly forcings for CONUS counties/states.",
        allow_abbrev=False)
    ap.add_argument("--project", default=None,
                    help="Earth Engine Cloud project id (required by current EE)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--start", type=int, default=START_YEAR)
    ap.add_argument("--end", type=int, default=END_YEAR)
    ap.add_argument("--freq", choices=("monthly", "annual"), default="monthly",
                    help="monthly (default) keeps one row per county-month, "
                         "which is what a convex water coefficient needs; "
                         "annual collapses to one row per county-year")
    ap.add_argument("--verbose", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.project, args.data_dir, args.start, args.end,
        args.freq, args.verbose)
    return 0


if __name__ == "__main__":
    main()