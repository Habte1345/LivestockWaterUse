"""
county_water_consumption.py

County-level water consumption (WC) for each livestock type:

    WC = WCC (L head-1 day-1) x head count

    <DATA_DIR>/
        county_level_WC_dairy_cattle.feather
        county_level_WC_beef_cattle.feather
        county_level_WC_hogs.feather
        county_level_WC_poultry.feather
        county_level_WC_all_livestock.feather   the four stacked

--------------------------------------------------------------------------
TWO THINGS THAT DECIDE WHETHER THE NUMBERS ARE RIGHT
--------------------------------------------------------------------------
1. THE MONTHS MULTIPLY BEFORE THEY SUM. WCC is a rate and varies by month,
   the head count does not, so the annual volume is

       sum_m ( WCC_m x head x days_in_month )

   not (annual mean WCC) x head x 365. The two differ because WCC is
   convex in temperature, and taking the mean first understates the summer
   peak -- by 4 to 9 percent depending on the seasonal range.

2. THE CENSUS ONLY COUNTS EVERY FIVE YEARS. Head counts exist for 2002,
   2007, 2012, 2017 and 2022. By default WC is produced for those years
   only, so every value rests on a counted inventory. --interpolate fills
   the intervening years by linear interpolation between censuses and
   labels each row in `head_origin`, so an interpolated count is never
   mistaken for a counted one. Extrapolation beyond the census span is
   NOT offered: a herd count carried back to 1985 would be an invention,
   and the 1985-2001 span is where livestock numbers moved most.

Missing values propagate rather than being filled. A county with no
climate has no WCC, a county whose count was withheld by NASS disclosure
rules has no head, and either way it has no WC -- reported, never zeroed.

Usage
-----
    python county_water_consumption.py
    python county_water_consumption.py --interpolate
    from county_water_consumption import run; run()
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:                                     # pragma: no cover
    class tqdm:
        def __init__(self, iterable=None, **kw):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or ())

        def set_postfix_str(self, s=""):
            pass

        def close(self):
            pass

        @staticmethod
        def write(msg):
            print(msg)

_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/wc_county"
WCC_DIR = f"{_REPO}/data/wcc_county"
USDA_DIR = f"{_REPO}/data/usda_census"

LIVESTOCK = ("dairy_cattle", "beef_cattle", "hogs", "poultry")

L_PER_M3 = 1000.0
L_PER_GAL = 3.785411784
MGAL_PER_L = 1.0 / (L_PER_GAL * 1e6)

OUT_COLS = ["year", "month", "fips", "state_fips", "county_name",
            "livestock_type", "head", "head_origin",
            "wcc_l_head_d", "wc_l_d", "wc_m3_d", "wc_mgal_d"]


class Report:
    def __init__(self, echo: bool = False):
        self.echo = echo

    def __call__(self, text: str = "") -> None:
        if self.echo:
            tqdm.write(text)


def find(base: Path, pattern: str) -> Path:
    hits = [p for p in base.glob(pattern) if "annual" not in p.name]
    if not hits:
        raise FileNotFoundError(f"no {pattern} in {base}")
    return sorted(hits)[-1]


def load_head(usda_dir: Path, name: str, interpolate: bool,
              rep: Report) -> pd.DataFrame:
    """
    County head counts. Census years as counted; intervening years by
    linear interpolation only if asked, never extrapolated beyond the
    census span.
    """
    df = pd.read_feather(find(usda_dir, f"usda_{name}_county_*.feather"))
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["head"] = pd.to_numeric(df["head"], errors="coerce")
    df = df.dropna(subset=["head"])
    df["head_origin"] = "census"

    census_years = sorted(int(y) for y in df["year"].unique())
    rep(f"  {name}: census years {census_years}, "
        f"{df['fips'].nunique():,} counties, {len(df):,} county-years")
    if not interpolate:
        return df[["year", "fips", "county_name", "head", "head_origin"]]

    lo, hi = min(census_years), max(census_years)
    span = np.arange(lo, hi + 1)
    out = []
    for fips, g in df.groupby("fips"):
        g = g.sort_values("year")
        if len(g) < 2:
            out.append(g)
            continue
        vals = np.interp(span, g["year"].to_numpy(float),
                         g["head"].to_numpy(float))
        out.append(pd.DataFrame({
            "year": span, "fips": fips,
            "county_name": g["county_name"].iloc[0],
            "head": vals,
            "head_origin": np.where(np.isin(span, g["year"]),
                                    "census", "interpolated"),
        }))
    res = pd.concat(out, ignore_index=True)
    n_i = int((res["head_origin"] == "interpolated").sum())
    rep(f"    interpolated to {lo}-{hi}: {n_i:,} of {len(res):,} county-years "
        f"filled between censuses; none extrapolated outside the span")
    return res


def run(data_dir: str = DATA_DIR, wcc_dir: str = WCC_DIR,
        usda_dir: str = USDA_DIR, interpolate: bool = False,
        verbose: bool = False) -> Dict[str, pd.DataFrame]:
    base = Path(data_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    wccp, usdap = Path(wcc_dir).expanduser(), Path(usda_dir).expanduser()
    rep = Report(echo=verbose)

    days = {m: calendar.monthrange(2001, m)[1] for m in range(1, 13)}
    outputs, stacked, summary = {}, [], []

    bar = tqdm(LIVESTOCK, desc="County WC", unit="type")
    for name in bar:
        bar.set_postfix_str(name)

        wcc = pd.read_feather(find(wccp, f"wcc_county_{name}_*.feather"))
        wcc["fips"] = wcc["fips"].astype(str).str.zfill(5)
        wcc = wcc.rename(columns={"wcc_ann_l_d": "wcc_l_head_d"})
        head = load_head(usdap, name, interpolate, rep)

        # Inner join: a row needs both a coefficient and a count.
        df = wcc.merge(head, on=["year", "fips"], how="inner")
        if df.empty:
            rep(f"  {name}: no overlapping year between WCC and census")
            continue

        df["wc_l_d"] = df["wcc_l_head_d"] * df["head"]
        df["wc_m3_d"] = df["wc_l_d"] / L_PER_M3
        df["wc_mgal_d"] = df["wc_l_d"] * MGAL_PER_L
        df["livestock_type"] = name
        for c in OUT_COLS:
            if c not in df:
                df[c] = np.nan

        df = df[OUT_COLS].sort_values(["year", "month", "fips"]) \
            .reset_index(drop=True)
        outputs[f"county_level_WC_{name}.feather"] = df
        stacked.append(df)

        n_nan = int(df["wc_l_d"].isna().sum())
        # Annual volume: months multiply first, then sum. Using the mean
        # WCC and 365 days would understate the summer peak.
        if "month" in df and df["month"].notna().any():
            vol = (df.assign(d=df["month"].map(days))
                   .eval("wc_l_d * d")
                   .groupby([df["year"], df["fips"]]).sum())
            natl = vol.groupby(level=0).sum() / 1e9      # billion L per year
        else:
            natl = pd.Series(dtype=float)

        summary.append({
            "livestock_type": name, "rows": len(df),
            "counties": df["fips"].nunique(),
            "years": f"{int(df['year'].min())}-{int(df['year'].max())}",
            "no_value": n_nan,
            "mean_wc_mgal_d": float(df["wc_mgal_d"].mean()),
        })
        rep(f"  {name}: {len(df):,} rows, {df['fips'].nunique():,} counties, "
            f"{int(df['year'].min())}-{int(df['year'].max())}"
            + (f", {n_nan:,} with no value" if n_nan else ""))
        for y, v in natl.items():
            rep(f"      {int(y)} national total {v:,.1f} billion L/yr "
                f"({v * 1e9 * MGAL_PER_L / 365:,.0f} Mgal/d average)")
    bar.close()

    if stacked:
        outputs["county_level_WC_all_livestock.feather"] = pd.concat(
            stacked, ignore_index=True)

    for fname, df in outputs.items():
        df.reset_index(drop=True).to_feather(base / fname)

    if not verbose:
        print("Done.")
        for s in summary:
            print(f"  {s['livestock_type']:<14}{s['rows']:>9,} rows  "
                  f"{s['counties']:>5,} counties  {s['years']:<10}"
                  f"mean {s['mean_wc_mgal_d']:>8.3f} Mgal/d"
                  + (f"   {s['no_value']:,} no value" if s["no_value"] else ""))
        print(f"  head counts  {'census + interpolated' if interpolate else 'census years only'}")
        print(f"  files        {len(outputs)}")
        print(f"  tables       {base}")
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="County water consumption = WCC x head count.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--wcc-dir", default=WCC_DIR)
    ap.add_argument("--usda-dir", default=USDA_DIR)
    ap.add_argument("--interpolate", action="store_true",
                    help="fill years between censuses by interpolation; "
                         "off by default so every count is a counted one")
    ap.add_argument("--verbose", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.wcc_dir, args.usda_dir,
        args.interpolate, args.verbose)
    return 0


if __name__ == "__main__":
    main()