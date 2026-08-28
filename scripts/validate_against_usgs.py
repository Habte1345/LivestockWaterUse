"""
validate_against_usgs.py

End-to-end test of the pathway. Water withdrawal is formed from the
modelled consumption and the downscaled ratio,

    WW = WC / ratio

and WC, WW and the ratio are each compared with the USGS county
observations wherever the years overlap.

    <DATA_DIR>/
        validation_ww_county.feather      modelled vs observed withdrawal
        validation_wc_county.feather      modelled vs observed consumption
        validation_ratio_county.feather   modelled vs observed ratio
        validation_metrics.feather        every comparison, one row each

--------------------------------------------------------------------------
WHAT CAN AND CANNOT BE TESTED
--------------------------------------------------------------------------
USGS stopped collecting county CONSUMPTIVE use after 1995. County
WITHDRAWAL continues to 2015. The modelled series starts in 2002, because
that is the first agricultural census with county head counts.

So the year overlaps are:

    withdrawal   2005, 2010, 2015     testable
    consumption  1985, 1990, 1995     NO overlap with the modelled series
    ratio        1985, 1990, 1995     testable, the ratio needs no head count

The consumption comparison is therefore reported as untestable rather than
manufactured by extrapolating head counts back to 1985. Livestock numbers
moved a great deal over 1985-2001, and a back-extrapolated count would
make the comparison a test of the extrapolation rather than of the model.

Modelled WC is summed across the four livestock types, which is what the
USGS livestock category covers.

SCORING. Withdrawal spans several orders of magnitude across counties, so
skill is reported both on the raw values, where a handful of large
counties dominate, and on log10, where every county counts roughly
equally. Both belong in the manuscript: the raw number says whether the
national total is right, the log number says whether the spatial pattern
is right.

Usage
-----
    python validate_against_usgs.py
    from validate_against_usgs import run; run()
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/validation"
WC_DIR = f"{_REPO}/data/wc_county"
RATIO_DIR = f"{_REPO}/data/ratio_county"
USGS_DIR = f"{_REPO}/data/usgs_ls_state_county_1960_2015"

LIVESTOCK = ("dairy_cattle", "beef_cattle", "hogs", "poultry")
MIN_RATIO = 0.05        # below this, WW = WC/ratio explodes


class Report:
    def __init__(self, echo: bool = True):
        self.echo = echo

    def __call__(self, text: str = "") -> None:
        if self.echo:
            print(text)


def metrics(obs, mod, label: str, log: bool = False) -> dict:
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    if log:
        ok &= (obs > 0) & (mod > 0)
        obs, mod = np.log10(obs[ok]), np.log10(mod[ok])
    else:
        obs, mod = obs[ok], mod[ok]
    if len(obs) < 3:
        return {"comparison": label, "n": int(len(obs))}
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    resid = mod - obs
    # Nash-Sutcliffe is identical to R2 about the 1:1 line here, and is the
    # form a hydrology reviewer will expect to see.
    return {
        "comparison": label,
        "n": int(len(obs)),
        "r2": 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot else np.nan,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "bias": float(np.mean(resid)),
        "pbias_pct": float(100 * np.sum(resid) / np.sum(obs))
        if np.sum(obs) else np.nan,
        "pearson_r": float(np.corrcoef(obs, mod)[0, 1]),
        "median_obs": float(np.median(obs)),
        "median_mod": float(np.median(mod)),
    }


def load_modelled_wc(wc_dir: Path, rep: Report) -> pd.DataFrame:
    """
    Modelled consumption as one row per county-year.

    Two aggregations, and they are NOT the same operation:

      across MONTHS      average -- WC is a rate in Mgal/d, so the annual
                         figure is the mean daily rate, not the sum of
                         twelve monthly rates
      across LIVESTOCK   sum -- the four types share a county, and USGS
                         reports them together as one livestock category

    Doing both with a single groupby sums the months as well, which
    inflates every county by a factor of twelve. That was the cause of the
    9x discrepancy against USGS.
    """
    frames = []
    for name in LIVESTOCK:
        p = wc_dir / f"county_level_WC_{name}.feather"
        if not p.exists():
            rep(f"  missing {p.name}")
            continue
        d = pd.read_feather(p)
        d["fips"] = d["fips"].astype(str).str.zfill(5)
        keys = ["year", "fips"]
        if "month" in d.columns:
            # Mean over months FIRST, within this livestock type.
            d = d.groupby(keys, as_index=False)["wc_mgal_d"].mean()
        else:
            d = d[keys + ["wc_mgal_d"]]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no county_level_WC_*.feather in {wc_dir}")
    # Then sum the four types.
    wc = (pd.concat(frames, ignore_index=True)
          .groupby(["year", "fips"], as_index=False)["wc_mgal_d"].sum())
    rep(f"  modelled WC: {len(wc):,} county-years, "
        f"{wc['fips'].nunique():,} counties, "
        f"{sorted(int(y) for y in wc['year'].unique())}")
    return wc


def load_modelled_ratio(ratio_dir: Path, rep: Report) -> pd.DataFrame:
    for f in ("county_level_ratios_all.feather",
              "county_level_ratios_dairy_cattle.feather"):
        p = ratio_dir / f
        if p.exists():
            d = pd.read_feather(p)
            d["fips"] = d["fips"].astype(str).str.zfill(5)
            d = d[["year", "fips", "wc_ww_ratio"]].drop_duplicates(
                ["year", "fips"])
            rep(f"  modelled ratio: {len(d):,} county-years from {f}")
            return d
    raise FileNotFoundError(f"no county_level_ratios_*.feather in {ratio_dir}")


def load_usgs(usgs_dir: Path, rep: Report) -> pd.DataFrame:
    hits = sorted(usgs_dir.glob("usgs_livestock_county_*.feather"))
    if not hits:
        raise FileNotFoundError(f"no usgs_livestock_county_*.feather in {usgs_dir}")
    d = pd.read_feather(hits[-1])
    d["fips"] = d["fips"].astype(str).str.zfill(5)
    keep = ["year", "fips"] + [c for c in
                               ("ls_withdrawal_fresh_mgd",
                                "ls_consumptive_fresh_mgd", "ls_cw_ratio")
                               if c in d.columns]
    d = d[keep]
    for col in keep[2:]:
        yrs = sorted(int(y) for y in d.loc[d[col].notna(), "year"].unique())
        rep(f"  USGS {col}: {int(d[col].notna().sum()):,} values, years {yrs}")
    return d


def run(data_dir: str = DATA_DIR, wc_dir: str = WC_DIR,
        ratio_dir: str = RATIO_DIR, usgs_dir: str = USGS_DIR,
        verbose: bool = True) -> Dict[str, pd.DataFrame]:
    base = Path(data_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    rep = Report(echo=verbose)

    rep("=" * 74)
    rep("INPUTS")
    rep("=" * 74)
    wc = load_modelled_wc(Path(wc_dir).expanduser(), rep)
    ratio = load_modelled_ratio(Path(ratio_dir).expanduser(), rep)
    usgs = load_usgs(Path(usgs_dir).expanduser(), rep)

    # WW = WC / ratio, on the modelled side.
    m = wc.merge(ratio, on=["year", "fips"], how="inner")
    n_small = int((m["wc_ww_ratio"] < MIN_RATIO).sum())
    if n_small:
        rep(f"\n  {n_small:,} county-years have a ratio below {MIN_RATIO}; "
            f"WW = WC/ratio is unstable there and is left as NaN")
    m.loc[m["wc_ww_ratio"] < MIN_RATIO, "wc_ww_ratio"] = np.nan
    m["ww_mgal_d"] = m["wc_mgal_d"] / m["wc_ww_ratio"]
    rep(f"  modelled WW: {int(m['ww_mgal_d'].notna().sum()):,} county-years")

    # WC and WW are limited to years with head counts; the RATIO is not --
    # it needs no census. Joining the ratio through the WC table would have
    # thrown away the 1985-1995 years where USGS actually published a
    # county ratio, which is the only place that comparison can be made.
    j = m.merge(usgs, on=["year", "fips"], how="inner")
    j_ratio = ratio.merge(usgs, on=["year", "fips"], how="inner")
    outputs, rows = {}, []

    rep("\n" + "=" * 74)
    rep("COMPARISONS")
    rep("=" * 74)

    checks = [
        ("withdrawal", "ls_withdrawal_fresh_mgd", "ww_mgal_d", True),
        ("consumption", "ls_consumptive_fresh_mgd", "wc_mgal_d", True),
        ("ratio", "ls_cw_ratio", "wc_ww_ratio", False),
    ]
    for label, ocol, mcol, use_log in checks:
        src = j_ratio if label == "ratio" else j
        if ocol not in src.columns:
            rep(f"\n  {label}: USGS column {ocol} absent -- not tested")
            continue
        sub = src[["year", "fips", ocol, mcol]].dropna()
        if sub.empty:
            oy = sorted(int(y) for y in
                        usgs.loc[usgs[ocol].notna(), "year"].unique())
            my = sorted(int(y) for y in src["year"].unique())
            rep(f"\n  {label}: NO OVERLAPPING YEARS -- not tested")
            rep(f"    USGS has {oy}")
            rep(f"    the model has {my}")
            rep(f"    Nothing is fabricated to close that gap.")
            rows.append({"comparison": label, "n": 0,
                         "note": "no overlapping years"})
            continue

        yrs = sorted(int(y) for y in sub["year"].unique())
        rep(f"\n  {label}: {len(sub):,} county-years, "
            f"{sub['fips'].nunique():,} counties, years {yrs}")
        s = metrics(sub[ocol], sub[mcol], label)
        rep(f"    raw    R2 {s.get('r2', float('nan')):+.4f}  "
            f"RMSE {s.get('rmse', float('nan')):.4f}  "
            f"PBIAS {s.get('pbias_pct', float('nan')):+.1f}%  "
            f"r {s.get('pearson_r', float('nan')):+.3f}")
        rows.append(s)
        if use_log:
            sl = metrics(sub[ocol], sub[mcol], f"{label} (log10)", log=True)
            rep(f"    log10  R2 {sl.get('r2', float('nan')):+.4f}  "
                f"RMSE {sl.get('rmse', float('nan')):.4f}  "
                f"r {sl.get('pearson_r', float('nan')):+.3f}")
            rows.append(sl)
            rep(f"    national total: observed "
                f"{sub[ocol].sum():,.0f} vs modelled {sub[mcol].sum():,.0f} "
                f"Mgal/d "
                f"({100*(sub[mcol].sum()/sub[ocol].sum() - 1):+.1f}%)")
        outputs[f"validation_{label[:5]}_county.feather"] = sub.rename(
            columns={ocol: "observed", mcol: "modelled"})

    metrics_df = pd.DataFrame(rows)
    outputs["validation_metrics.feather"] = metrics_df
    for fname, df in outputs.items():
        df.reset_index(drop=True).to_feather(base / fname)

    rep("\n" + "=" * 74)
    rep("SUMMARY")
    rep("=" * 74)
    if not metrics_df.empty and "r2" in metrics_df:
        rep(metrics_df[[c for c in ("comparison", "n", "r2", "rmse",
                                    "pbias_pct", "pearson_r")
                        if c in metrics_df]].to_string(index=False))
    rep(f"\n  tables  {base}")
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate county WC, WW and ratio against USGS.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--wc-dir", default=WC_DIR)
    ap.add_argument("--ratio-dir", default=RATIO_DIR)
    ap.add_argument("--usgs-dir", default=USGS_DIR)
    ap.add_argument("--quiet", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.wc_dir, args.ratio_dir, args.usgs_dir,
        not args.quiet)
    return 0


if __name__ == "__main__":
    main()