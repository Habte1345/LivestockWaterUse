"""
ratio_anomaly_downscale.py

County water-consumption to water-withdrawal ratio, built as

    county_ratio = state_ratio x g(county climate - state climate)

instead of predicting the ratio from absolute climate.

    <DATA_DIR>/
        county_level_ratios_<livestock>.feather
        county_level_ratios_all.feather
        ratio_anomaly_performance.feather
        ratio_anomaly_folds.feather
        ratio_anomaly_oof.feather

--------------------------------------------------------------------------
WHY THIS DIFFERS FROM ratio_ann_downscale.py
--------------------------------------------------------------------------
That script trains on state-level rows and applies the result to counties.
Every training row is a whole state, so the model only ever sees BETWEEN-
state climate differences, and is then asked about WITHIN-state differences
between counties. Those are different regimes and the training data says
nothing about the second one.

It also throws away the single most informative fact available. Knowing
only which state a row belongs to explains about 62 percent of the ratio;
predicting the absolute ratio from absolute climate reached 17 percent in
sample and 8 percent out of fold. Discarding a 62 percent baseline to chase
a 17 percent signal is the wrong way round.

Here the state ratio is kept as the base, and climate is asked to do the
one job it can do: explain why a county deviates from its own state. The
multiplier g is trained on the USGS COUNTY ratios, which is the only place
within-state variation is actually observed.

    target      the county ratio itself
    predictors  county climate minus its state mean, same year, PLUS the
                state ratio the county sits in

The county ratio is the target directly, not a log anomaly. The state
ratio then enters as a predictor rather than as a fixed multiplier, so the
model can use it and adjust it instead of being locked to it. That keeps
the 62 percent that state identity already explains available to the
model while still letting climate move the answer.

VALIDATION. Grouped by STATE, so a state never appears in both halves --
the model must generalise to states it has not seen, not merely to other
counties of a state it has memorised. A year-held-out split is reported
alongside, since the intended application extrapolates in time.

BASELINE. Setting g = 1, that is, giving every county its state's ratio
unchanged, is the honest thing to beat. If climate cannot beat it, the
state ratio applied uniformly is the correct product and this script says
so rather than dressing up a worse answer.

Usage
-----
    python ratio_anomaly_downscale.py
    python ratio_anomaly_downscale.py --folds 8 --hidden 8
    from ratio_anomaly_downscale import run; run()
"""

from __future__ import annotations

import argparse
import contextlib
import warnings as _warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@contextlib.contextmanager
def quiet():
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore")
        yield


_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/ratio_county"
USGS_DIR = f"{_REPO}/data/usgs_ls_state_county_1960_2015"
CLIMATE_DIR = f"{_REPO}/data/climate"

LIVESTOCK = ("dairy_cattle", "beef_cattle", "hogs", "poultry")
FORCINGS = {"Temp": "temp_c", "RH": "rh_pct",
            "Wind": "wind2_kmh", "Sunlight": "sunlight_h"}
# The state ratio is a predictor, not a multiplier: with the county ratio
# as the direct target the model needs to know the level it is predicting
# around, and letting it weight that itself beats forcing it.
ANOM_COLS = ["d_temp_c", "d_rh_pct", "d_wind_kmh", "d_sunlight_h"]
FEATURES = ["state_ratio"] + ANOM_COLS

RATIO_CAP = 1.0          # physical bound: consumption cannot exceed withdrawal


class Report:
    def __init__(self, echo: bool = False):
        self.echo = echo

    def __call__(self, text: str = "") -> None:
        if self.echo:
            tqdm.write(text)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def climate_anomalies(climate_dir: Path, rep: Report) -> pd.DataFrame:
    """
    County climate expressed as a deviation from its own state, same year.

    The deviation is the predictor, not the absolute value: the state
    ratio already carries the absolute climate through whatever produced
    it, and what remains to explain is the within-state contrast.
    """
    out = None
    for stem, col in FORCINGS.items():
        hits = sorted(climate_dir.glob(f"{stem}_county_level_*.feather"))
        if not hits:
            raise FileNotFoundError(f"no {stem}_county_level_*.feather")
        df = pd.read_feather(hits[-1])
        cc, cs = f"{col}_county", f"{col}_state"
        if cc not in df or cs not in df:
            raise KeyError(f"{hits[-1].name} lacks {cc} or {cs}")
        sub = (df[["year", "fips", "state_fips", cc, cs]]
               .groupby(["year", "fips", "state_fips"], as_index=False).mean())
        name = "wind_kmh" if col == "wind2_kmh" else col
        sub[f"d_{name}"] = sub[cc] - sub[cs]
        sub = sub[["year", "fips", "state_fips", f"d_{name}"]]
        out = sub if out is None else out.merge(
            sub, on=["year", "fips", "state_fips"], how="inner")

    out["fips"] = out["fips"].astype(str).str.zfill(5)
    out["state_fips"] = out["state_fips"].astype(str).str.zfill(2)
    rep(f"  climate anomalies: {len(out):,} county-years, "
        f"{out['fips'].nunique():,} counties")
    for f in ANOM_COLS:
        rep(f"    {f:<14} sd {out[f].std():.3f}  "
            f"range [{out[f].min():+.2f}, {out[f].max():+.2f}]")
    return out


def load_ratios(usgs_dir: Path, rep: Report) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sp = sorted(usgs_dir.glob("usgs_livestock_state_*.feather"))
    cp = sorted(usgs_dir.glob("usgs_livestock_county_*.feather"))
    if not cp:
        raise FileNotFoundError(
            f"no usgs_livestock_county_*.feather in {usgs_dir} -- this design "
            f"trains on the county ratios, which is where within-state "
            f"variation is observed")
    ct = pd.read_feather(cp[-1])[["year", "fips", "ls_cw_ratio"]].dropna()
    ct["fips"] = ct["fips"].astype(str).str.zfill(5)
    ct["state_fips"] = ct["fips"].str[:2]

    st = pd.DataFrame()
    if sp:
        st = pd.read_feather(sp[-1])[["year", "state_fips", "ls_cw_ratio"]].dropna()
        st["state_fips"] = st["state_fips"].astype(str).str.zfill(2)

    n_over = int((ct["ls_cw_ratio"] > RATIO_CAP).sum())
    if n_over:
        rep(f"  capped {n_over} county ratio(s) above {RATIO_CAP} "
            f"(max {ct['ls_cw_ratio'].max():.2f}) -- source errors, the "
            f"ratio cannot exceed 1")
        ct["ls_cw_ratio"] = ct["ls_cw_ratio"].clip(upper=RATIO_CAP)
    rep(f"  county ratios: {len(ct):,} county-years, years "
        f"{sorted(int(y) for y in ct['year'].unique())}")
    return st, ct


def state_base(ct: pd.DataFrame, st: pd.DataFrame, rep: Report,
               all_years: np.ndarray) -> pd.DataFrame:
    """
    The state ratio each county is scaled from, for EVERY year the climate
    covers -- not only the years USGS published one.

    This is the piece that decides whether the model can produce anything
    outside the observation years. The state base is built from the
    published series where it exists (1960-1980) and from the mean of a
    state's own counties where the county file covers the year
    (1985-1995). USGS collected no consumptive use after 1995, so beyond
    that the state's last observed ratio is CARRIED FORWARD.

    Carrying forward is an assumption and is labelled as one in
    `base_source`, so a 2015 value is never mistaken for an observed one.
    It is the honest option: without it the model returns nothing after
    1995, which is exactly the period the dataset is for.
    """
    agg = (ct.groupby(["year", "state_fips"], as_index=False)["ls_cw_ratio"]
           .mean().rename(columns={"ls_cw_ratio": "state_ratio"}))
    agg["base_source"] = "mean of the state's counties"

    if not st.empty:
        s = st.rename(columns={"ls_cw_ratio": "published"})
        s["published"] = s["published"].clip(upper=RATIO_CAP)
        agg = agg.merge(s, on=["year", "state_fips"], how="outer")
        use = agg["published"].notna()
        agg.loc[use, "state_ratio"] = agg.loc[use, "published"]
        agg.loc[use, "base_source"] = "published USGS state ratio"
        agg = agg.drop(columns=["published"])

    # Expand to every climate year, per state, then carry the nearest
    # observed value into the gaps.
    states = agg["state_fips"].dropna().unique()
    grid = pd.MultiIndex.from_product(
        [np.sort(all_years).astype(int), states],
        names=["year", "state_fips"]).to_frame(index=False)
    full = grid.merge(agg, on=["year", "state_fips"], how="left")
    full = full.sort_values(["state_fips", "year"])

    observed = full["state_ratio"].notna()
    full["state_ratio"] = full.groupby("state_fips")["state_ratio"].ffill()
    full["state_ratio"] = full.groupby("state_fips")["state_ratio"].bfill()
    filled = full["state_ratio"].notna() & ~observed
    full.loc[filled, "base_source"] = "carried from the nearest observed year"

    n_obs, n_fill = int(observed.sum()), int(filled.sum())
    rep(f"  state base: {n_obs:,} observed, {n_fill:,} carried forward or "
        f"back, {int(full['state_ratio'].isna().sum()):,} still missing")
    rep(f"    sources: "
        f"{dict(full['base_source'].value_counts(dropna=False))}")
    return full.dropna(subset=["state_ratio"])


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def candidates(hidden, seed, max_iter):
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV
    common = dict(activation="relu", solver="adam", learning_rate_init=1e-3,
                  max_iter=max_iter, early_stopping=True, n_iter_no_change=25,
                  validation_fraction=0.2, random_state=seed)
    return {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: RidgeCV(alphas=np.logspace(-3, 3, 25)),
        f"ann{tuple(hidden)}": lambda: MLPRegressor(
            hidden_layer_sizes=tuple(hidden), alpha=1e-2, **common),
        "ann(8,)": lambda: MLPRegressor(hidden_layer_sizes=(8,),
                                        alpha=1e-1, **common),
    }


def evaluate(rep: Report, d: pd.DataFrame, hidden, seed, max_iter,
             folds: int, group_col: str, label: str):
    """
    Grouped cross-validation of the multiplier g.

    The target is the county ratio directly, so predictions need no
    reconstruction -- they are only clipped to the physical bound.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_squared_error

    X = d[FEATURES].to_numpy(float)
    y = d["ls_cw_ratio"].to_numpy(float)
    obs = y
    base = d["state_ratio"].to_numpy(float)
    groups = d[group_col].to_numpy()

    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(folds, n_groups))
    models = candidates(hidden, seed, max_iter)
    preds = {k: np.full(len(y), np.nan) for k in models}
    rows = []

    bar = tqdm(list(gkf.split(X, y, groups)),
               desc=f"  CV by {label}", unit="fold", leave=True)
    for k, (tr, va) in enumerate(bar, start=1):
        sc = StandardScaler().fit(X[tr])
        row = {"fold": k, "n_train": len(tr), "n_val": len(va),
               "n_groups_val": int(len(np.unique(groups[va])))}
        for name, make in models.items():
            with quiet():
                m = make()
                m.fit(sc.transform(X[tr]) if name.startswith("ann") else X[tr],
                      y[tr])
                pv = m.predict(sc.transform(X[va]) if name.startswith("ann")
                               else X[va])
            preds[name][va] = pv
            rec = np.clip(pv, 1e-6, RATIO_CAP)
            row[f"rmse_{name}"] = float(np.sqrt(
                mean_squared_error(obs[va], rec)))
        row["rmse_state_base"] = float(np.sqrt(
            mean_squared_error(obs[va], base[va])))
        rows.append(row)
        bar.set_postfix_str(" | ".join(
            f"{n} {row[f'rmse_{n}']:.4f}" for n in models)
            + f" | base {row['rmse_state_base']:.4f}")
    bar.close()

    stats = {"n_samples": len(y), "n_folds": gkf.get_n_splits(),
             "grouping": label}
    rec_all = {}
    for name in models:
        rec = np.clip(preds[name], 1e-6, RATIO_CAP)
        rec_all[name] = rec
        stats[f"cv_rmse_{name}"] = float(np.sqrt(mean_squared_error(obs, rec)))
        stats[f"cv_r2_{name}"] = r2_score(obs, rec)
    stats["cv_rmse_state_base"] = float(np.sqrt(mean_squared_error(obs, base)))
    stats["cv_r2_state_base"] = r2_score(obs, base)

    ranked = sorted(models, key=lambda n: stats[f"cv_rmse_{n}"])
    best = ranked[0]
    stats["best_model"] = best
    stats["models"] = "|".join(models)
    stats["beats_state_base"] = bool(
        stats[f"cv_rmse_{best}"] < stats["cv_rmse_state_base"])

    rep(f"\n  cross-validated by {label}, scored on the county ratio")
    for name in ranked:
        mark = "  <- best" if name == best else ""
        rep(f"    {name:<12} R2 {stats[f'cv_r2_{name}']:+.4f}  "
            f"RMSE {stats[f'cv_rmse_{name}']:.4f}{mark}")
    rep(f"    {'state base':<12} R2 {stats['cv_r2_state_base']:+.4f}  "
        f"RMSE {stats['cv_rmse_state_base']:.4f}   <- the thing to beat")
    if not stats["beats_state_base"]:
        rep("    Climate does NOT improve on the state ratio applied as is.")

    oof = pd.DataFrame({"observed": obs, "state_base": base,
                        "predicted": rec_all[best], "group": groups,
                        "year": d["year"].to_numpy()})
    return stats, pd.DataFrame(rows), oof, models[best]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(data_dir: str = DATA_DIR, usgs_dir: str = USGS_DIR,
        climate_dir: str = CLIMATE_DIR, hidden: Tuple[int, ...] = (8,),
        seed: int = 42, max_iter: int = 600, folds: int = 8,
        verbose: bool = False) -> Dict[str, pd.DataFrame]:
    base_dir = Path(data_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    rep = Report(echo=verbose)

    anom = climate_anomalies(Path(climate_dir).expanduser(), rep)
    st, ct = load_ratios(Path(usgs_dir).expanduser(), rep)
    sbase = state_base(ct, st, rep, anom["year"].unique())

    d = (ct.merge(sbase, on=["year", "state_fips"], how="inner")
         .merge(anom, on=["year", "fips", "state_fips"], how="inner")
         .dropna(subset=FEATURES + ["ls_cw_ratio", "state_ratio"]))
    d = d[(d["ls_cw_ratio"] > 0) & (d["state_ratio"] > 0)]
    if d.empty:
        raise RuntimeError("no overlap between county ratios and climate")

    rep(f"\n  training rows: {len(d):,} county-years, "
        f"{d['fips'].nunique():,} counties, {d['state_fips'].nunique()} states")
    rep(f"  target: county ratio directly, min {d['ls_cw_ratio'].min():.3f} "
        f"median {d['ls_cw_ratio'].median():.3f} "
        f"max {d['ls_cw_ratio'].max():.3f}")
    rep(f"    {100*np.isclose(d['ls_cw_ratio'], 1.0).mean():.1f}% of counties "
        f"sit at exactly 1.0")

    stats_s, folds_s, oof_s, _ = evaluate(rep, d, hidden, seed, max_iter,
                                          folds, "state_fips", "state")
    stats_y, folds_y, _, _ = evaluate(rep, d, hidden, seed, max_iter,
                                      min(folds, d["year"].nunique()),
                                      "year", "year")

    # Final model: whichever won the state-held-out comparison, refitted on
    # everything, then applied to every county-year that has climate.
    from sklearn.preprocessing import StandardScaler
    best = stats_s["best_model"]
    sc = StandardScaler().fit(d[FEATURES].to_numpy(float))
    with quiet():
        model = candidates(hidden, seed, max_iter)[best]()
        Xf = (sc.transform(d[FEATURES].to_numpy(float))
              if best.startswith("ann") else d[FEATURES].to_numpy(float))
        model.fit(Xf, d["ls_cw_ratio"].to_numpy(float))

    allc = anom.merge(sbase, on=["year", "state_fips"], how="left")
    ok = allc[FEATURES].notna().all(axis=1) & allc["state_ratio"].notna()
    Xa = allc.loc[ok, FEATURES].to_numpy(float)
    pv = model.predict(sc.transform(Xa) if best.startswith("ann") else Xa)
    allc["wc_ww_ratio"] = np.nan
    allc.loc[ok, "wc_ww_ratio"] = np.clip(pv, 1e-6, RATIO_CAP)

    out = allc[["year", "fips", "state_fips", "state_ratio",
                "wc_ww_ratio", "base_source"]]
    rep(f"\n  county ratios: {int(out['wc_ww_ratio'].notna().sum()):,} values, "
        f"median {out['wc_ww_ratio'].median():.4f}, "
        f"{int((out['wc_ww_ratio'] > RATIO_CAP).sum()):,} above {RATIO_CAP}")

    outputs: Dict[str, pd.DataFrame] = {}
    for name in LIVESTOCK:
        dd = out.copy()
        dd["livestock_type"] = name
        outputs[f"county_level_ratios_{name}.feather"] = dd
    outputs["county_level_ratios_all.feather"] = out
    outputs["ratio_anomaly_performance.feather"] = pd.DataFrame(
        [stats_s, stats_y])
    outputs["ratio_anomaly_folds.feather"] = pd.concat(
        [folds_s.assign(grouping="state"), folds_y.assign(grouping="year")],
        ignore_index=True)
    outputs["ratio_anomaly_oof.feather"] = oof_s

    for fname, df in outputs.items():
        df.reset_index(drop=True).to_feather(base_dir / fname)

    if not verbose:
        print("Done.")
        print(f"  training      {len(d):,} county-years, "
              f"{d['fips'].nunique():,} counties, "
              f"{d['state_fips'].nunique()} states")
        for stats in (stats_s, stats_y):
            print(f"\n  held out by {stats['grouping']}, "
                  f"scored on the county ratio")
            for n in sorted(stats["models"].split("|"),
                            key=lambda n: stats[f"cv_rmse_{n}"]):
                mark = "  <- best" if n == stats["best_model"] else ""
                print(f"    {n:<12} R2 {stats[f'cv_r2_{n}']:+.4f}  "
                      f"RMSE {stats[f'cv_rmse_{n}']:.4f}{mark}")
            print(f"    {'state base':<12} R2 "
                  f"{stats['cv_r2_state_base']:+.4f}  "
                  f"RMSE {stats['cv_rmse_state_base']:.4f}   <- to beat")
        print()
        if stats_s["beats_state_base"] and stats_y["beats_state_base"]:
            print("  Climate anomalies improve on the state ratio under both "
                  "splits.")
        else:
            print("  Climate anomalies do NOT reliably improve on the state "
                  "ratio.")
            print("  The defensible product is then the state ratio applied "
                  "to its counties,")
            print("  with this comparison reported as the reason.")
        print(f"\n  county ratios "
              f"{int(out['wc_ww_ratio'].notna().sum()):,} values, "
              f"median {out['wc_ww_ratio'].median():.4f}")
        print(f"  tables        {base_dir}")
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="County ratio as state ratio times a climate-anomaly "
                    "multiplier.", allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--usgs-dir", default=USGS_DIR)
    ap.add_argument("--climate-dir", default=CLIMATE_DIR)
    ap.add_argument("--hidden", type=int, nargs="+", default=[8])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.usgs_dir, args.climate_dir, tuple(args.hidden),
        args.seed, args.max_iter, args.folds, args.verbose)
    return 0


if __name__ == "__main__":
    main()