"""
wcc_ann_downscale.py

Pathway 1: transfer the generic, MLR-derived water consumption coefficient
to county level using county-specific climate.

    generic WCC          the MLR equations evaluated over sampled animal
                         physiology and climate -- one value per sample,
                         not tied to any place
    ANN                  learns WCC from the CLIMATE columns of those
                         samples alone
    county WCC           that network applied to observed county climate

    <DATA_DIR>/
        wcc_county_<type>_<start>_<end>.feather      county-month WCC
        wcc_county_<type>_annual_<start>_<end>.feather
        wcc_ann_performance.feather                  fit statistics

--------------------------------------------------------------------------
HOW THE TRANSFER WORKS, AND WHAT IT CAN AND CANNOT DO
--------------------------------------------------------------------------
Each surrogate sample carries both physiology (body weight, intake, age,
lactation) and climate (temperature, humidity, wind, sunlight). The MLR
maps all of them to a WCC. The ANN is then trained on the CLIMATE columns
only, with that WCC as target.

Physiology is therefore not dropped -- it shaped every target value. What
the ANN sees is the spread of WCC that physiology produces at a given
climate, and it learns the centre of that spread. That is exactly what
"generic" means here: a coefficient appropriate to a county with a typical
herd, conditioned on that county's climate.

Two consequences worth stating in the Methods rather than discovering later:

  * The ANN cannot exceed what the MLR encodes. Any climate signal it finds
    was put there by the MLR; there is no additional information in the
    training data. The R2 below is a measure of how well the network
    reproduces the MLR, not of how well either predicts reality.

  * Residual scatter is irreducible BY CONSTRUCTION. At one temperature the
    target takes many values because physiology varied, and no climate-only
    input can distinguish them. The ceiling is reported alongside R2 so a
    modest R2 is not mistaken for a poor network.

For a linear MLR the quantity the ANN converges to also has a closed form,
so the analytic transfer is computed as well and the two are compared. If
they agree, the pathway is confirmed from two directions; if they do not,
the gap is ANN fitting error and is visible rather than hidden.

--------------------------------------------------------------------------
Usage
-----
    python wcc_ann_downscale.py
    python wcc_ann_downscale.py --n-samples 50000 --hidden 64 32
    from wcc_ann_downscale import run; run()
"""

from __future__ import annotations

import argparse
import sys
import zlib
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
DATA_DIR = f"{_REPO}/data/wcc_county"
WCC_DIR = f"{_REPO}/data/wcc_mlr"
CLIMATE_DIR = f"{_REPO}/data/climate"
USDA_DIR = f"{_REPO}/data/usda_census"

LIVESTOCK = ("dairy_cattle", "beef_cattle", "hogs", "poultry")

# Climate predictors available at county level from the ERA5 tables. These
# are the ANN inputs. Precipitation is deliberately excluded: no intake
# equation depends on it, so including it would only give the network room
# to fit noise.
CLIMATE_FEATURES = ["temp_c", "rh_pct", "wind_kmh", "sunlight_h"]

# Herd composition that the USDA census reports per county. These are
# OBSERVED county attributes, not sampled physiology, so they belong on the
# input side. Leaving them out was what capped hogs and poultry: for those
# two the composition contributes as much WCC variance as climate does, and
# a climate-only model can then do no better than predict the conditional
# mean -- which is the flat horizontal cloud in the scatter.
#
# The key is the sample column each maps onto, so training and application
# share one feature space.
COMPOSITION_FEATURES = {
    "dairy_cattle": {},
    "beef_cattle": {},
    "hogs": {"breeding_fraction": "breeding"},
    "poultry": {"layer_fraction": "layer"},
}

# The census file each fraction is read from.
COMPOSITION_SOURCE = {"hogs": "usda_hogs_county_*.feather",
                      "poultry": "usda_poultry_county_*.feather"}


def county_features(name: str) -> List[str]:
    """Everything the ANN sees: climate plus any observed composition."""
    return CLIMATE_FEATURES + list(COMPOSITION_FEATURES[name].values())

# Mapping from the ERA5 county table column names to the sample column
# names, so training and application share one feature space.
ERA5_RENAME = {"wind2_kmh": "wind_kmh"}

N_SAMPLES = 50_000
SEED = 42
L_PER_GAL = 3.785411784
GAL_PER_L = 1.0 / L_PER_GAL


import contextlib
import warnings as _warnings


@contextlib.contextmanager
def warnings_suppressed():
    """
    partial_fit emits a convergence warning on every single call, because
    each call is one epoch and one epoch never converges. Suppressed only
    around the fit step, so genuine warnings elsewhere still surface.
    """
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=UserWarning)
        try:
            from sklearn.exceptions import ConvergenceWarning
            _warnings.filterwarnings("ignore", category=ConvergenceWarning)
        except ImportError:
            pass
        yield


class Report:
    def __init__(self, echo: bool = False):
        self.echo = echo

    def __call__(self, text: str = "") -> None:
        if self.echo:
            tqdm.write(text)


# ---------------------------------------------------------------------------
# generic surrogate targets
# ---------------------------------------------------------------------------

def climate_response(wcc_mod, name: str, df: pd.DataFrame) -> np.ndarray:
    """
    Climate multiplier: the published intake response at each sample's own
    climate, divided by the same response at the reference climate.

    THIS IS WHAT MAKES THE TRANSFER POSSIBLE. The MLR is fitted on
    physiological predictors alone, with the intake equations evaluated at
    REF_CLIMATE, so its prediction is a GENERIC coefficient that does not
    move with weather -- by construction, not by oversight. Training a
    climate-only network directly on that prediction is asking it to learn
    a function of variables the target does not contain, and it can only
    return the mean.

    The response functions themselves are climate-dependent and already
    live in wcc_mlr: Murphy's dairy equation carries minimum temperature,
    Winchester's beef equation an exponential in temperature, and the swine
    and poultry relations are water-to-feed ratios that rise with heat
    load. Evaluating each at the sampled climate and at the reference, and
    taking the ratio, isolates the part of the response the MLR deliberately
    held fixed.

    The MLR equations are untouched: the generic level is theirs, and this
    supplies only the relative adjustment around it.
    """
    R = wcc_mod.REF_CLIMATE
    t = df["temp_c"].to_numpy(dtype=float)

    if name == "dairy_cattle":
        # Murphy et al. (1983) takes MINIMUM temperature, which is neither
        # an ANN input nor available in the county climate tables. It is
        # therefore derived from the mean using the sampler's mean diurnal
        # range, so the response is a function of mean temperature -- the
        # variable the network and the county data both carry. Using the
        # sampled minimum directly would make part of the response depend
        # on a quantity the model can never observe.
        tmin = t - 11.0 / 2.0
        # Intake and milk yield at their REPRESENTATIVE values, not per
        # sample. Murphy's equation adds temperature rather than
        # multiplying by it, so a per-sample ratio still carries that
        # sample's intake and yield: two counties with identical climate
        # would receive different multipliers according to physiology the
        # county data cannot supply, and the transfer fans out at both ends
        # of the range. The other three species use water-to-feed ratios,
        # in which physiology cancels exactly, which is why they do not.
        # Evaluating at the mean matches the single generic coefficient the
        # multiplier is applied to.
        dmi = float(np.nanmean(df["dmi_kg_d"].to_numpy(dtype=float)))
        milk = float(np.nanmean(df["milk_kg_d"].to_numpy(dtype=float)))
        num = wcc_mod.dairy_fwi_murphy(dmi, milk, 60.0, tmin)
        den = wcc_mod.dairy_fwi_murphy(dmi, milk, 60.0, R["temp_min_c"])
    elif name == "beef_cattle":
        # Winchester and Morris (1956), the intake-driven beef relation.
        dmi = df["dmi_kg_d"].to_numpy(dtype=float)
        num = wcc_mod.beef_wi_winchester(dmi, t)
        den = wcc_mod.beef_wi_winchester(dmi, R["temp_c"])
    elif name == "hogs":
        num = wcc_mod.swine_water_feed_ratio(t)
        den = wcc_mod.swine_water_feed_ratio(R["temp_c"])
    else:                                            # poultry mixture
        layer = df["layer"].to_numpy(dtype=float)
        wfr_n = (layer * wcc_mod.layer_water_feed_ratio(t)
                 + (1.0 - layer) * 1.77)
        wfr_d = (layer * wcc_mod.layer_water_feed_ratio(R["temp_c"])
                 + (1.0 - layer) * 1.77)
        num = wfr_n * wcc_mod.broiler_coefficient(t)
        den = wfr_d * wcc_mod.broiler_coefficient(R["temp_c"])

    den = np.where(np.abs(den) > 1e-12, den, np.nan)
    return np.asarray(num, dtype=float) / np.asarray(den, dtype=float)


def make_surrogate(wcc_mod, name: str, coefs: dict, n: int,
                   seed: int, broiler_fraction: float) -> pd.DataFrame:
    """
    Generic WCC samples: physiology and climate sampled together, then the
    trained MLR equation evaluated on them.

    Physiology is sampled over the same literature-constrained ranges the
    MLR was fitted on, so the targets span the full physiological spread
    rather than a single representative animal. That spread is what the
    ANN averages over.
    """
    rng = np.random.default_rng(seed + zlib.crc32(name.encode("utf-8")) % 100_000)
    df = (wcc_mod.GENERATORS[name](rng, n, broiler_fraction)
          if name == "poultry" else wcc_mod.GENERATORS[name](rng, n))
    df = wcc_mod.add_interactions(name, df)

    # The generic coefficient from the fitted MLR equation, taken at
    # REPRESENTATIVE physiology -- a single value per livestock type.
    #
    # A county has a climate but no physiology: the census reports how many
    # animals there are, not their body weight or intake. The quantity
    # transferred to a county is therefore one generic coefficient per
    # species, adjusted for that county's climate. Carrying the per-sample
    # physiological scatter into the target instead would add variance that
    # no county-level input can resolve, and the network would be scored
    # against noise it has no means of predicting.
    per_sample = wcc_mod.predict_wcc(coefs, name, df)
    generic = float(np.nanmean(per_sample))
    df["wcc_mlr_generic_l_d"] = generic
    df["wcc_physiology_l_d"] = per_sample

    # Scaled by the published climate response, so the target varies with
    # weather exactly as the source equations say it should.
    expected = generic * climate_response(wcc_mod, name, df)
    df["wcc_expected_generic_l_d"] = expected

    # Plus the between-herd residual that wcc_mlr already quantifies:
    # spread in management, genetics and housing that no climate variable
    # carries. Without it the target is an exact function of the four
    # inputs the network receives, so it can be fitted to machine
    # precision. An R2 of 1.00 obtained that way is arithmetic, not skill,
    # and would misrepresent the transfer as perfect when it is merely
    # deterministic. With the residual present the network estimates the
    # conditional mean of a genuinely noisy quantity, which is the task it
    # performs when applied to a real county.
    cv = wcc_mod.RESIDUAL_CV[name]
    sigma = np.sqrt(np.log1p(cv ** 2))
    rng = np.random.default_rng(seed + 500_000
                                + zlib.crc32(name.encode("utf-8")) % 100_000)
    df["wcc_generic_l_d"] = expected * rng.lognormal(
        -0.5 * sigma ** 2, sigma, len(df))
    return df


def climate_dependent(wcc_mod, name: str, samples: pd.DataFrame) -> Dict[str, bool]:
    """
    Which model features depend on something the county table supplies.

    Determined empirically by permuting the county-observable columns and
    seeing which features change, rather than by parsing names. Name
    matching is what made an earlier version treat `temp_sq` -- a pure
    climate term -- as an interaction, because the county table has no
    column with that literal name.
    """
    rng = np.random.default_rng(0)
    shuffled = samples.copy()
    idx = rng.permutation(len(samples))
    for c in county_features(name):
        if c in shuffled:
            shuffled[c] = samples[c].to_numpy()[idx]
    shuffled = wcc_mod.add_interactions(name, shuffled)

    out = {}
    for f in wcc_mod.FEATURES[name]:
        if f not in samples or f not in shuffled:
            out[f] = False
            continue
        out[f] = not np.allclose(samples[f].to_numpy(dtype=float),
                                 shuffled[f].to_numpy(dtype=float),
                                 equal_nan=True)
    return out


def analytic_transfer(wcc_mod, name: str, coefs: dict,
                      samples: pd.DataFrame, county: pd.DataFrame,
                      n_bins: int = 60) -> np.ndarray:
    """
    Closed-form county WCC, for comparison with the ANN.

    The MLR is linear, so

        E[WCC | observed] = b0 + sum_j b_j E[f_j | observed]

    The subtlety is what "E[f_j | observed]" means. Originally physiology
    was sampled independently of everything the county reports, so the
    plug-in sample MEAN was exact. That stopped being true once herd
    composition became a county input: hog body weight is now built from
    the breeding share, and poultry intake from the layer share, so
    E[dmi] is no longer E[dmi | breeding].

    Using the unconditional mean there was a real error -- it biased the
    closed form enough that the ANN appeared to beat the exact conditional
    mean, which is impossible. Physiology is therefore conditioned on the
    composition variable by binning the samples on it and taking per-bin
    means, then mapping each county onto its bin. Climate remains
    independent of physiology, so it needs no such treatment.
    """
    dep = climate_dependent(wcc_mod, name, samples)
    obs = county_features(name)
    comp = list(COMPOSITION_FEATURES[name].values())

    frame = pd.DataFrame(index=range(len(county)))
    for c in obs:
        if c in county:
            frame[c] = county[c].to_numpy(dtype=float)

    phys = [c for c in samples.columns
            if c not in obs and samples[c].dtype.kind in "fiub"]

    edges, lab, cl = np.array([]), None, None
    if comp:
        # Physiology conditioned on the composition the county reports.
        key = comp[0]
        edges = np.unique(np.quantile(samples[key],
                                      np.linspace(0, 1, n_bins + 1)))
        if len(edges) > 2:
            lab = np.clip(np.digitize(samples[key], edges[1:-1]),
                          0, len(edges) - 2)
            table = (samples[phys].groupby(lab).mean()
                     .reindex(range(len(edges) - 1)).ffill().bfill())
            cl = np.clip(np.digitize(frame[key].to_numpy(dtype=float),
                                     edges[1:-1]), 0, len(edges) - 2)
            for c in phys:
                frame[c] = table[c].to_numpy()[cl]
        else:
            for c in phys:
                frame[c] = float(samples[c].mean())
    else:
        for c in phys:
            frame[c] = float(samples[c].mean())

    frame = wcc_mod.add_interactions(name, frame)

    # A feature that depends on no county input must take the sample mean
    # of the FEATURE, not be rebuilt from mean inputs. For a nonlinear term
    # those differ: mean(bw)^2 is not mean(bw^2), and the gap is the
    # variance. That error put a constant offset on every beef prediction,
    # because the beef model carries a bw_sq term.
    #
    # Where composition is a county input, that mean is taken within the
    # composition bin, for the same conditioning reason as above.
    indep = [f for f in wcc_mod.FEATURES[name] if not dep.get(f, False)]
    indep_mean = {}
    if indep:
        if comp and len(edges) > 2:
            ftab = (samples[indep].groupby(lab).mean()
                    .reindex(range(len(edges) - 1)).ffill().bfill())
            for f in indep:
                indep_mean[f] = ftab[f].to_numpy()[cl]
        else:
            for f in indep:
                indep_mean[f] = np.full(len(county), float(samples[f].mean()))

    out = np.full(len(county), float(coefs["intercept"]))
    for f in wcc_mod.FEATURES[name]:
        b = float(coefs[f])
        if dep.get(f, False) and f in frame:
            out += b * frame[f].to_numpy(dtype=float)
        else:
            out += b * indep_mean[f]

    # Match the ANN: no predictors, no value.
    out[~complete_mask(county, obs)] = np.nan
    return out


# ---------------------------------------------------------------------------
# ANN
# ---------------------------------------------------------------------------

def train_ann(rep: Report, name: str, samples: pd.DataFrame,
              hidden: Tuple[int, ...], seed: int, max_iter: int,
              patience: int = 25
              ) -> Tuple[object, object, dict, pd.DataFrame, pd.DataFrame]:
    """
    Train with an explicit epoch loop so progress is visible.

    sklearn's MLPRegressor.fit() runs to completion with no per-epoch hook,
    so a long fit shows nothing at all until it returns. Driving it with
    partial_fit one epoch at a time costs nothing and makes the training
    and validation curves observable as they happen.

    Early stopping and best-weight restoration are handled here rather than
    by sklearn's early_stopping, because that carves its own validation
    split out of the training data and does not expose the curve.
    """
    import copy
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    feats = county_features(name)
    X = samples[feats].to_numpy(dtype=float)
    y = samples["wcc_generic_l_d"].to_numpy(dtype=float)

    # Three-way split: train, validation for early stopping, and a test set
    # that neither the fitting nor the stopping rule ever sees.
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.2,
                                            random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.2,
                                          random_state=seed)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = (scaler.transform(a) for a in (Xtr, Xva, Xte))

    net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                       solver="adam", learning_rate_init=1e-3,
                       max_iter=1, warm_start=True, early_stopping=False,
                       random_state=seed)

    best_val, best_state, best_epoch, since = np.inf, None, 0, 0
    history = []

    bar = tqdm(range(1, max_iter + 1), desc=f"  training {name}",
               unit="epoch", leave=False)
    for epoch in bar:
        with warnings_suppressed():
            net.partial_fit(Xtr_s, ytr)

        tr_pred, va_pred = net.predict(Xtr_s), net.predict(Xva_s)
        tr_rmse = float(np.sqrt(mean_squared_error(ytr, tr_pred)))
        va_rmse = float(np.sqrt(mean_squared_error(yva, va_pred)))
        va_r2 = float(r2_score(yva, va_pred))
        history.append({"livestock_type": name, "epoch": epoch,
                        "train_rmse_l_d": tr_rmse, "val_rmse_l_d": va_rmse,
                        "val_r2": va_r2})

        bar.set_postfix_str(
            f"train {tr_rmse:.3f} | val {va_rmse:.3f} | val R2 {va_r2:.4f}")

        if va_rmse < best_val - 1e-9:
            best_val, best_epoch, since = va_rmse, epoch, 0
            best_state = copy.deepcopy((net.coefs_, net.intercepts_))
        else:
            since += 1
            if since >= patience:
                bar.set_postfix_str(
                    f"early stop at {epoch}, best epoch {best_epoch}")
                break
    bar.close()

    if best_state is not None:
        net.coefs_, net.intercepts_ = best_state

    pred = net.predict(Xte_s)

    # Per-sample predictions at the BEST weights, kept so the transfer can
    # be plotted against the surrogate target afterwards without retraining.
    preds = pd.concat([
        pd.DataFrame({"livestock_type": name, "split": "train",
                      "wcc_surrogate_l_d": ytr,
                      "wcc_ann_l_d": net.predict(Xtr_s)}),
        pd.DataFrame({"livestock_type": name, "split": "validation",
                      "wcc_surrogate_l_d": yva,
                      "wcc_ann_l_d": net.predict(Xva_s)}),
        pd.DataFrame({"livestock_type": name, "split": "test",
                      "wcc_surrogate_l_d": yte,
                      "wcc_ann_l_d": pred}),
    ], ignore_index=True)

    # Ceiling: the best any climate-only model could do, given that
    # physiology moves the target at fixed climate. Estimated by binning on
    # temperature and taking the within-bin mean as the best possible
    # prediction. It bins on one variable only, so it understates what a
    # four-variable model can reach -- treat it as a floor on the ceiling.
    bins = pd.qcut(pd.Series(Xte[:, 0]), 40, duplicates="drop")
    within = pd.Series(yte).groupby(bins, observed=True).transform("mean")
    ceiling = r2_score(yte, within)

    stats = {
        "livestock_type": name,
        "n_samples": len(samples),
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "r2_test": r2_score(yte, pred),
        "rmse_test_l_d": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2_ceiling_climate_only": ceiling,
        "pct_of_ceiling": float(100 * r2_score(yte, pred) / ceiling)
        if ceiling > 0 else np.nan,
        "features": ",".join(feats),
        "hidden_layers": str(hidden),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_rmse_l_d": best_val,
    }

    rep(f"\n  {name}")
    rep(f"    ANN {hidden}, {stats['epochs_run']} epochs, best at "
        f"{best_epoch} (val RMSE {best_val:.4f} L/d)")
    rep(f"    R2 test {stats['r2_test']:.4f} | climate-only ceiling "
        f"{ceiling:.4f} ({stats['pct_of_ceiling']:.1f}% of it reached)")
    rep(f"    RMSE test {stats['rmse_test_l_d']:.4f} L/d")
    rep("    the ceiling is below 1 by construction: at one climate the "
        "target")
    rep("    takes many values because physiology varied, and no "
        "climate-only")
    rep("    input can separate them.")
    return net, scaler, stats, pd.DataFrame(history), preds


# ---------------------------------------------------------------------------
# county application
# ---------------------------------------------------------------------------

def load_county_climate(climate_dir: Path, rep: Report) -> pd.DataFrame:
    """
    Assemble the county climate predictors from the per-forcing tables.
    Every table shares the same keys, so they join cleanly on fips.
    """
    stems = {"Temp": "temp_c_county", "RH": "rh_pct_county",
             "Wind": "wind2_kmh_county", "Sunlight": "sunlight_h_county"}
    out = None
    for stem, col in stems.items():
        hits = sorted(climate_dir.glob(f"{stem}_county_level_*.feather"))
        if not hits:
            raise FileNotFoundError(
                f"no {stem}_county_level_*.feather in {climate_dir}. "
                f"Run era5_forcings.py first.")
        df = pd.read_feather(hits[-1])
        keys = [k for k in ("year", "month", "fips", "state_fips")
                if k in df.columns]
        sub = df[keys + [col]].rename(
            columns={col: col.replace("_county", "")})
        out = sub if out is None else out.merge(sub, on=keys, how="inner")

    out = out.rename(columns=ERA5_RENAME)
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    rep(f"  county climate: {len(out):,} rows, "
        f"{out['fips'].nunique():,} counties, "
        f"{out['year'].min()}-{out['year'].max()}")

    # Missing values are carried through as NaN rather than dropped or
    # imputed, so a county with no climate ends up with no WCC instead of a
    # fabricated one. Reported per variable because the cause differs:
    # a county too small to hold an ERA5 pixel centre loses every variable,
    # whereas sunlight_h alone goes missing where the radiation ratio
    # cannot be inverted.
    miss = {c: int(out[c].isna().sum()) for c in CLIMATE_FEATURES
            if c in out.columns}
    bad = {k: v for k, v in miss.items() if v}
    if bad:
        rep(f"  missing values by variable: {bad}")
        n_rows = int(out[CLIMATE_FEATURES].isna().any(axis=1).sum())
        n_cty = int(out.loc[out[CLIMATE_FEATURES].isna().any(axis=1),
                            "fips"].nunique())
        rep(f"  -> {n_rows:,} rows ({100*n_rows/len(out):.2f}%) across "
            f"{n_cty:,} counties will receive no WCC")
    else:
        rep("  no missing climate values")
    return out


def load_composition(usda_dir: Path, name: str, years: np.ndarray,
                     rep: Report) -> Optional[pd.DataFrame]:
    """
    County herd composition, interpolated from the census years onto every
    climate year.

    The census reports 2002, 2007, 2012, 2017 and 2022. A composition
    FRACTION changes far more slowly and smoothly than a head count, so
    linear interpolation between census years is defensible where a
    head-count interpolation would not be. Outside the census span the
    nearest observed value is held constant, and every row is labelled with
    how it was obtained so an extrapolated value is never mistaken for an
    observed one.
    """
    pattern = COMPOSITION_SOURCE.get(name)
    if pattern is None:
        return None
    hits = sorted(usda_dir.glob(pattern))
    if not hits:
        rep(f"  {name}: no {pattern} in {usda_dir}; composition unavailable")
        return None

    col = [c for c in COMPOSITION_FEATURES[name]][0]
    target = COMPOSITION_FEATURES[name][col]
    df = pd.read_feather(hits[-1])
    if col not in df.columns:
        rep(f"  {name}: {hits[-1].name} has no {col}; composition unavailable")
        return None

    df = df[["year", "fips", col]].dropna()
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    census_years = sorted(df["year"].unique())

    out = []
    for fips, g in df.groupby("fips"):
        g = g.sort_values("year")
        vals = np.interp(years, g["year"].to_numpy(dtype=float),
                         g[col].to_numpy(dtype=float))
        origin = np.where(
            years < g["year"].min(), "held from earliest census",
            np.where(years > g["year"].max(), "held from latest census",
                     "interpolated between census years"))
        out.append(pd.DataFrame({"fips": fips, "year": years,
                                 target: vals,
                                 f"{target}_origin": origin}))
    res = pd.concat(out, ignore_index=True)

    n_in = int((res[f"{target}_origin"] == "interpolated between census years").sum())
    rep(f"  {name}: {col} for {res['fips'].nunique():,} counties, census "
        f"years {[int(y) for y in census_years]}")
    rep(f"    {n_in:,} of {len(res):,} county-years interpolated within the "
        f"census span; the rest held from the nearest census year")
    return res


def coverage_check(rep: Report, name: str, samples: pd.DataFrame,
                   county: pd.DataFrame) -> dict:
    """
    How much county climate falls outside the range the ANN was trained on.

    A neural network extrapolates arbitrarily, without warning, so this is
    the number that decides whether a county prediction can be trusted.
    """
    rep(f"\n  {name}: county climate against the training range")
    frac_out = {}
    for f in county_features(name):
        if f not in county.columns:
            rep(f"    {f:<12} NOT PRESENT in the county table")
            frac_out[f] = 1.0
            continue
        lo, hi = samples[f].min(), samples[f].max()
        v = county[f].dropna()
        if v.empty:
            rep(f"    {f:<12} no finite county values")
            frac_out[f] = 1.0
            continue
        out = float(((v < lo) | (v > hi)).mean())
        frac_out[f] = out
        flag = "   <-- EXTRAPOLATING" if out > 0.01 else ""
        rep(f"    {f:<12} train [{lo:8.2f}, {hi:8.2f}]  county "
            f"[{v.min():8.2f}, {v.max():8.2f}]  outside {100*out:5.2f}%{flag}")
    return frac_out


def complete_mask(county: pd.DataFrame, feats: List[str]) -> np.ndarray:
    """Rows where every predictor the model needs is finite."""
    have = [f for f in feats if f in county.columns]
    if len(have) < len(feats):
        return np.zeros(len(county), dtype=bool)
    return np.isfinite(county[have].to_numpy(dtype=float)).all(axis=1)


def apply_to_counties(net, scaler, county: pd.DataFrame,
                      feats: List[str]) -> np.ndarray:
    """
    Predict on complete rows only.

    A neural network cannot take NaN, and imputing a climate value would
    invent a county's weather. Incomplete rows stay NaN in the output so
    that a missing county is visibly missing downstream.
    """
    out = np.full(len(county), np.nan)
    ok = complete_mask(county, feats)
    if ok.any():
        X = county.loc[ok, feats].to_numpy(dtype=float)
        out[ok] = net.predict(scaler.transform(X))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(data_dir: str = DATA_DIR, wcc_dir: str = WCC_DIR,
        climate_dir: str = CLIMATE_DIR, usda_dir: str = USDA_DIR,
        n_samples: int = N_SAMPLES,
        seed: int = SEED, hidden: Tuple[int, ...] = (64, 32),
        max_iter: int = 400, broiler_fraction: float = 0.78,
        patience: int = 25, verbose: bool = False
        ) -> Dict[str, pd.DataFrame]:
    base = Path(data_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    rep = Report(echo=verbose)

    # The MLR module supplies both the samplers and the fitted equations,
    # so the surrogate targets here are the same construction that produced
    # wcc_mlr_coefficients.feather.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib
    wcc_mod = importlib.import_module("wcc_mlr")

    coef_path = Path(wcc_dir).expanduser() / "wcc_mlr_coefficients.feather"
    if not coef_path.exists():
        raise FileNotFoundError(
            f"{coef_path} not found. Run wcc_mlr.py first.")
    coef_df = pd.read_feather(coef_path)

    county_climate = load_county_climate(Path(climate_dir).expanduser(), rep)
    usda_path = Path(usda_dir).expanduser()
    all_years = np.sort(county_climate["year"].unique()).astype(float)
    monthly = "month" in county_climate.columns
    y0, y1 = int(county_climate["year"].min()), int(county_climate["year"].max())

    outputs: Dict[str, pd.DataFrame] = {}
    all_stats, histories, predictions = [], [], []

    bar = tqdm(LIVESTOCK, desc="Downscaling WCC", unit="type")
    for name in bar:
        bar.set_postfix_str(name)
        row = coef_df[coef_df["livestock_type"] == name]
        if row.empty:
            rep(f"  {name}: no fitted equation, skipped")
            continue
        coefs = row.iloc[0].to_dict()

        # Attach observed county composition where the census reports it.
        comp = load_composition(usda_path, name, all_years, rep)
        county_in = county_climate
        if comp is not None:
            county_in = county_climate.merge(
                comp, on=["fips", "year"], how="left")

        samples = make_surrogate(wcc_mod, name, coefs, n_samples, seed,
                                 broiler_fraction)
        net, scaler, stats, hist, preds = train_ann(
            rep, name, samples, hidden, seed, max_iter, patience)
        histories.append(hist)
        predictions.append(preds)
        cov = coverage_check(rep, name, samples, county_in)
        stats.update({f"frac_outside_{k}": v for k, v in cov.items()})
        all_stats.append(stats)

        feats = county_features(name)
        out = county_in[[c for c in
                         ("year", "month", "fips", "state_fips")
                         if c in county_in.columns]].copy()
        out["wcc_ann_l_d"] = apply_to_counties(net, scaler, county_in, feats)
        out["wcc_analytic_l_d"] = analytic_transfer(
            wcc_mod, name, coefs, samples, county_in)
        for extra in COMPOSITION_FEATURES[name].values():
            if extra in county_in:
                out[extra] = county_in[extra].to_numpy()
                oc = f"{extra}_origin"
                if oc in county_in:
                    out[oc] = county_in[oc].to_numpy()
        out["wcc_ann_gal_d"] = out["wcc_ann_l_d"] * GAL_PER_L
        out["livestock_type"] = name

        both = out[["wcc_ann_l_d", "wcc_analytic_l_d"]].dropna()
        rep(f"\n  {name}: ANN against the closed-form transfer")
        if len(both) > 1:
            d = both["wcc_ann_l_d"] - both["wcc_analytic_l_d"]
            denom = both["wcc_analytic_l_d"].abs().replace(0, np.nan)
            rep(f"    mean abs difference {d.abs().mean():.4f} L/d "
                f"({100 * (d.abs() / denom).mean():.2f}% of the analytic value)")
            rep(f"    correlation "
                f"{np.corrcoef(both['wcc_ann_l_d'], both['wcc_analytic_l_d'])[0,1]:.6f}")
            all_stats[-1]["ann_vs_analytic_mae_l_d"] = float(d.abs().mean())
        else:
            rep("    not computable: no rows with complete climate")
            all_stats[-1]["ann_vs_analytic_mae_l_d"] = np.nan

        n_nan = int(out["wcc_ann_l_d"].isna().sum())
        if n_nan:
            rep(f"    {n_nan:,} rows ({100*n_nan/len(out):.2f}%) left as NaN "
                f"for want of complete county climate")
        all_stats[-1]["n_county_rows_no_climate"] = n_nan

        n_neg = int((out["wcc_ann_l_d"] < 0).sum())
        if n_neg:
            rep(f"    NOTE: {n_neg:,} negative predictions "
                f"({100*n_neg/len(out):.3f}%), reported not clipped")

        outputs[f"wcc_county_{name}_{y0}_{y1}.feather"] = out

        if monthly:
            keys = ["year", "fips", "state_fips"]
            # mean skips NaN months; a county-year with no valid month
            # stays NaN rather than becoming zero.
            ann = (out.groupby(keys, as_index=False)
                   .agg(wcc_ann_l_d=("wcc_ann_l_d", "mean"),
                        wcc_ann_gal_d=("wcc_ann_gal_d", "mean"),
                        wcc_analytic_l_d=("wcc_analytic_l_d", "mean"),
                        n_months=("wcc_ann_l_d", "count")))
            ann["livestock_type"] = name
            outputs[f"wcc_county_{name}_annual_{y0}_{y1}.feather"] = ann
    bar.close()

    outputs["wcc_ann_performance.feather"] = pd.DataFrame(all_stats)
    if histories:
        # Per-epoch curves, so the training run can be plotted afterwards
        # rather than only summarised.
        outputs["wcc_ann_training_history.feather"] = pd.concat(
            histories, ignore_index=True)
    if predictions:
        outputs["wcc_ann_predictions.feather"] = pd.concat(
            predictions, ignore_index=True)

    for fname, df in outputs.items():
        df.reset_index(drop=True).to_feather(base / fname)

    if not verbose:
        print("Done.")
        for s in all_stats:
            print(f"  {s['livestock_type']:<14}R2 {s['r2_test']:.4f}  "
                  f"(ceiling {s['r2_ceiling_climate_only']:.4f}, "
                  f"{s['pct_of_ceiling']:.0f}% reached)  "
                  f"{s['epochs_run']:>3} epochs, best {s['best_epoch']:>3}  "
                  f"vs closed form "
                  f"{s.get('ann_vs_analytic_mae_l_d', float('nan')):.4f} L/d")
        print(f"  granularity  {'county-month' if monthly else 'county-year'}")
        print(f"  files        {len(outputs)} written")
        print(f"  tables       {base}")
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Transfer generic MLR WCC to county level with an ANN.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--wcc-dir", default=WCC_DIR)
    ap.add_argument("--climate-dir", default=CLIMATE_DIR)
    ap.add_argument("--usda-dir", default=USDA_DIR,
                    help="census tables holding the herd composition")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--max-iter", type=int, default=400,
                    help="maximum training epochs per livestock type")
    ap.add_argument("--patience", type=int, default=25,
                    help="epochs without validation improvement before "
                         "stopping")
    ap.add_argument("--broiler-fraction", type=float, default=0.78)
    ap.add_argument("--verbose", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.wcc_dir, args.climate_dir, args.usda_dir,
        args.n_samples, args.seed, tuple(args.hidden), args.max_iter,
        args.broiler_fraction, args.patience, args.verbose)
    return 0


if __name__ == "__main__":
    main()













# """
# wcc_ann_downscale.py

# Pathway 1: transfer the generic, MLR-derived water consumption coefficient
# to county level using county-specific climate.

#     generic WCC          the MLR equations evaluated over sampled animal
#                          physiology and climate -- one value per sample,
#                          not tied to any place
#     ANN                  learns WCC from the CLIMATE columns of those
#                          samples alone
#     county WCC           that network applied to observed county climate

#     <DATA_DIR>/
#         wcc_county_<type>_<start>_<end>.feather      county-month WCC
#         wcc_county_<type>_annual_<start>_<end>.feather
#         wcc_ann_performance.feather                  fit statistics

# --------------------------------------------------------------------------
# HOW THE TRANSFER WORKS, AND WHAT IT CAN AND CANNOT DO
# --------------------------------------------------------------------------
# Each surrogate sample carries both physiology (body weight, intake, age,
# lactation) and climate (temperature, humidity, wind, sunlight). The MLR
# maps all of them to a WCC. The ANN is then trained on the CLIMATE columns
# only, with that WCC as target.

# Physiology is therefore not dropped -- it shaped every target value. What
# the ANN sees is the spread of WCC that physiology produces at a given
# climate, and it learns the centre of that spread. That is exactly what
# "generic" means here: a coefficient appropriate to a county with a typical
# herd, conditioned on that county's climate.

# Two consequences worth stating in the Methods rather than discovering later:

#   * The ANN cannot exceed what the MLR encodes. Any climate signal it finds
#     was put there by the MLR; there is no additional information in the
#     training data. The R2 below is a measure of how well the network
#     reproduces the MLR, not of how well either predicts reality.

#   * Residual scatter is irreducible BY CONSTRUCTION. At one temperature the
#     target takes many values because physiology varied, and no climate-only
#     input can distinguish them. The ceiling is reported alongside R2 so a
#     modest R2 is not mistaken for a poor network.

# For a linear MLR the quantity the ANN converges to also has a closed form,
# so the analytic transfer is computed as well and the two are compared. If
# they agree, the pathway is confirmed from two directions; if they do not,
# the gap is ANN fitting error and is visible rather than hidden.

# --------------------------------------------------------------------------
# Usage
# -----
#     python wcc_ann_downscale.py
#     python wcc_ann_downscale.py --n-samples 50000 --hidden 64 32
#     from wcc_ann_downscale import run; run()
# """

# from __future__ import annotations

# import argparse
# import sys
# import zlib
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd

# try:
#     from tqdm.auto import tqdm
# except ImportError:                                     # pragma: no cover
#     class tqdm:
#         def __init__(self, iterable=None, total=None, desc=None, **kw):
#             self.iterable = iterable

#         def __iter__(self):
#             return iter(self.iterable or ())

#         def update(self, n=1):
#             pass

#         def set_postfix_str(self, s=""):
#             pass

#         def close(self):
#             pass

#         @staticmethod
#         def write(msg):
#             print(msg)

# _REPO = "/scratch/hdagne1/LivestockWaterUse"
# DATA_DIR = f"{_REPO}/data/wcc_county"
# WCC_DIR = f"{_REPO}/data/wcc_mlr"
# CLIMATE_DIR = f"{_REPO}/data/climate"
# USDA_DIR = f"{_REPO}/data/usda_census"

# LIVESTOCK = ("dairy_cattle", "beef_cattle", "hogs", "poultry")

# # Climate predictors available at county level from the ERA5 tables. These
# # are the ANN inputs. Precipitation is deliberately excluded: no intake
# # equation depends on it, so including it would only give the network room
# # to fit noise.
# CLIMATE_FEATURES = ["temp_c", "rh_pct", "wind_kmh", "sunlight_h"]

# # Herd composition that the USDA census reports per county. These are
# # OBSERVED county attributes, not sampled physiology, so they belong on the
# # input side. Leaving them out was what capped hogs and poultry: for those
# # two the composition contributes as much WCC variance as climate does, and
# # a climate-only model can then do no better than predict the conditional
# # mean -- which is the flat horizontal cloud in the scatter.
# #
# # The key is the sample column each maps onto, so training and application
# # share one feature space.
# COMPOSITION_FEATURES = {
#     "dairy_cattle": {},
#     "beef_cattle": {},
#     "hogs": {"breeding_fraction": "breeding"},
#     "poultry": {"layer_fraction": "layer"},
# }

# # The census file each fraction is read from.
# COMPOSITION_SOURCE = {"hogs": "usda_hogs_county_*.feather",
#                       "poultry": "usda_poultry_county_*.feather"}


# def county_features(name: str) -> List[str]:
#     """Everything the ANN sees: climate plus any observed composition."""
#     return CLIMATE_FEATURES + list(COMPOSITION_FEATURES[name].values())

# # Mapping from the ERA5 county table column names to the sample column
# # names, so training and application share one feature space.
# ERA5_RENAME = {"wind2_kmh": "wind_kmh"}

# N_SAMPLES = 50_000
# SEED = 42
# L_PER_GAL = 3.785411784
# GAL_PER_L = 1.0 / L_PER_GAL


# import contextlib
# import warnings as _warnings


# @contextlib.contextmanager
# def warnings_suppressed():
#     """
#     partial_fit emits a convergence warning on every single call, because
#     each call is one epoch and one epoch never converges. Suppressed only
#     around the fit step, so genuine warnings elsewhere still surface.
#     """
#     with _warnings.catch_warnings():
#         _warnings.filterwarnings("ignore", category=UserWarning)
#         try:
#             from sklearn.exceptions import ConvergenceWarning
#             _warnings.filterwarnings("ignore", category=ConvergenceWarning)
#         except ImportError:
#             pass
#         yield


# class Report:
#     def __init__(self, echo: bool = False):
#         self.echo = echo

#     def __call__(self, text: str = "") -> None:
#         if self.echo:
#             tqdm.write(text)


# # ---------------------------------------------------------------------------
# # generic surrogate targets
# # ---------------------------------------------------------------------------

# def make_surrogate(wcc_mod, name: str, coefs: dict, n: int,
#                    seed: int, broiler_fraction: float) -> pd.DataFrame:
#     """
#     Generic WCC samples: physiology and climate sampled together, then the
#     trained MLR equation evaluated on them.

#     Physiology is sampled over the same literature-constrained ranges the
#     MLR was fitted on, so the targets span the full physiological spread
#     rather than a single representative animal. That spread is what the
#     ANN averages over.
#     """
#     rng = np.random.default_rng(seed + zlib.crc32(name.encode("utf-8")) % 100_000)
#     df = (wcc_mod.GENERATORS[name](rng, n, broiler_fraction)
#           if name == "poultry" else wcc_mod.GENERATORS[name](rng, n))
#     df = wcc_mod.add_interactions(name, df)
#     df["wcc_generic_l_d"] = wcc_mod.predict_wcc(coefs, name, df)
#     return df


# def climate_dependent(wcc_mod, name: str, samples: pd.DataFrame) -> Dict[str, bool]:
#     """
#     Which model features depend on something the county table supplies.

#     Determined empirically by permuting the county-observable columns and
#     seeing which features change, rather than by parsing names. Name
#     matching is what made an earlier version treat `temp_sq` -- a pure
#     climate term -- as an interaction, because the county table has no
#     column with that literal name.
#     """
#     rng = np.random.default_rng(0)
#     shuffled = samples.copy()
#     idx = rng.permutation(len(samples))
#     for c in county_features(name):
#         if c in shuffled:
#             shuffled[c] = samples[c].to_numpy()[idx]
#     shuffled = wcc_mod.add_interactions(name, shuffled)

#     out = {}
#     for f in wcc_mod.FEATURES[name]:
#         if f not in samples or f not in shuffled:
#             out[f] = False
#             continue
#         out[f] = not np.allclose(samples[f].to_numpy(dtype=float),
#                                  shuffled[f].to_numpy(dtype=float),
#                                  equal_nan=True)
#     return out


# def analytic_transfer(wcc_mod, name: str, coefs: dict,
#                       samples: pd.DataFrame, county: pd.DataFrame,
#                       n_bins: int = 60) -> np.ndarray:
#     """
#     Closed-form county WCC, for comparison with the ANN.

#     The MLR is linear, so

#         E[WCC | observed] = b0 + sum_j b_j E[f_j | observed]

#     The subtlety is what "E[f_j | observed]" means. Originally physiology
#     was sampled independently of everything the county reports, so the
#     plug-in sample MEAN was exact. That stopped being true once herd
#     composition became a county input: hog body weight is now built from
#     the breeding share, and poultry intake from the layer share, so
#     E[dmi] is no longer E[dmi | breeding].

#     Using the unconditional mean there was a real error -- it biased the
#     closed form enough that the ANN appeared to beat the exact conditional
#     mean, which is impossible. Physiology is therefore conditioned on the
#     composition variable by binning the samples on it and taking per-bin
#     means, then mapping each county onto its bin. Climate remains
#     independent of physiology, so it needs no such treatment.
#     """
#     dep = climate_dependent(wcc_mod, name, samples)
#     obs = county_features(name)
#     comp = list(COMPOSITION_FEATURES[name].values())

#     frame = pd.DataFrame(index=range(len(county)))
#     for c in obs:
#         if c in county:
#             frame[c] = county[c].to_numpy(dtype=float)

#     phys = [c for c in samples.columns
#             if c not in obs and samples[c].dtype.kind in "fiub"]

#     edges, lab, cl = np.array([]), None, None
#     if comp:
#         # Physiology conditioned on the composition the county reports.
#         key = comp[0]
#         edges = np.unique(np.quantile(samples[key],
#                                       np.linspace(0, 1, n_bins + 1)))
#         if len(edges) > 2:
#             lab = np.clip(np.digitize(samples[key], edges[1:-1]),
#                           0, len(edges) - 2)
#             table = (samples[phys].groupby(lab).mean()
#                      .reindex(range(len(edges) - 1)).ffill().bfill())
#             cl = np.clip(np.digitize(frame[key].to_numpy(dtype=float),
#                                      edges[1:-1]), 0, len(edges) - 2)
#             for c in phys:
#                 frame[c] = table[c].to_numpy()[cl]
#         else:
#             for c in phys:
#                 frame[c] = float(samples[c].mean())
#     else:
#         for c in phys:
#             frame[c] = float(samples[c].mean())

#     frame = wcc_mod.add_interactions(name, frame)

#     # A feature that depends on no county input must take the sample mean
#     # of the FEATURE, not be rebuilt from mean inputs. For a nonlinear term
#     # those differ: mean(bw)^2 is not mean(bw^2), and the gap is the
#     # variance. That error put a constant offset on every beef prediction,
#     # because the beef model carries a bw_sq term.
#     #
#     # Where composition is a county input, that mean is taken within the
#     # composition bin, for the same conditioning reason as above.
#     indep = [f for f in wcc_mod.FEATURES[name] if not dep.get(f, False)]
#     indep_mean = {}
#     if indep:
#         if comp and len(edges) > 2:
#             ftab = (samples[indep].groupby(lab).mean()
#                     .reindex(range(len(edges) - 1)).ffill().bfill())
#             for f in indep:
#                 indep_mean[f] = ftab[f].to_numpy()[cl]
#         else:
#             for f in indep:
#                 indep_mean[f] = np.full(len(county), float(samples[f].mean()))

#     out = np.full(len(county), float(coefs["intercept"]))
#     for f in wcc_mod.FEATURES[name]:
#         b = float(coefs[f])
#         if dep.get(f, False) and f in frame:
#             out += b * frame[f].to_numpy(dtype=float)
#         else:
#             out += b * indep_mean[f]

#     # Match the ANN: no predictors, no value.
#     out[~complete_mask(county, obs)] = np.nan
#     return out


# # ---------------------------------------------------------------------------
# # ANN
# # ---------------------------------------------------------------------------

# def train_ann(rep: Report, name: str, samples: pd.DataFrame,
#               hidden: Tuple[int, ...], seed: int, max_iter: int,
#               patience: int = 25
#               ) -> Tuple[object, object, dict, pd.DataFrame, pd.DataFrame]:
#     """
#     Train with an explicit epoch loop so progress is visible.

#     sklearn's MLPRegressor.fit() runs to completion with no per-epoch hook,
#     so a long fit shows nothing at all until it returns. Driving it with
#     partial_fit one epoch at a time costs nothing and makes the training
#     and validation curves observable as they happen.

#     Early stopping and best-weight restoration are handled here rather than
#     by sklearn's early_stopping, because that carves its own validation
#     split out of the training data and does not expose the curve.
#     """
#     import copy
#     from sklearn.neural_network import MLPRegressor
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import r2_score, mean_squared_error

#     feats = county_features(name)
#     X = samples[feats].to_numpy(dtype=float)
#     y = samples["wcc_generic_l_d"].to_numpy(dtype=float)

#     # Three-way split: train, validation for early stopping, and a test set
#     # that neither the fitting nor the stopping rule ever sees.
#     Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.2,
#                                             random_state=seed)
#     Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.2,
#                                           random_state=seed)

#     scaler = StandardScaler().fit(Xtr)
#     Xtr_s, Xva_s, Xte_s = (scaler.transform(a) for a in (Xtr, Xva, Xte))

#     net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
#                        solver="adam", learning_rate_init=1e-3,
#                        max_iter=1, warm_start=True, early_stopping=False,
#                        random_state=seed)

#     best_val, best_state, best_epoch, since = np.inf, None, 0, 0
#     history = []

#     bar = tqdm(range(1, max_iter + 1), desc=f"  training {name}",
#                unit="epoch", leave=False)
#     for epoch in bar:
#         with warnings_suppressed():
#             net.partial_fit(Xtr_s, ytr)

#         tr_pred, va_pred = net.predict(Xtr_s), net.predict(Xva_s)
#         tr_rmse = float(np.sqrt(mean_squared_error(ytr, tr_pred)))
#         va_rmse = float(np.sqrt(mean_squared_error(yva, va_pred)))
#         va_r2 = float(r2_score(yva, va_pred))
#         history.append({"livestock_type": name, "epoch": epoch,
#                         "train_rmse_l_d": tr_rmse, "val_rmse_l_d": va_rmse,
#                         "val_r2": va_r2})

#         bar.set_postfix_str(
#             f"train {tr_rmse:.3f} | val {va_rmse:.3f} | val R2 {va_r2:.4f}")

#         if va_rmse < best_val - 1e-9:
#             best_val, best_epoch, since = va_rmse, epoch, 0
#             best_state = copy.deepcopy((net.coefs_, net.intercepts_))
#         else:
#             since += 1
#             if since >= patience:
#                 bar.set_postfix_str(
#                     f"early stop at {epoch}, best epoch {best_epoch}")
#                 break
#     bar.close()

#     if best_state is not None:
#         net.coefs_, net.intercepts_ = best_state

#     pred = net.predict(Xte_s)

#     # Per-sample predictions at the BEST weights, kept so the transfer can
#     # be plotted against the surrogate target afterwards without retraining.
#     preds = pd.concat([
#         pd.DataFrame({"livestock_type": name, "split": "train",
#                       "wcc_surrogate_l_d": ytr,
#                       "wcc_ann_l_d": net.predict(Xtr_s)}),
#         pd.DataFrame({"livestock_type": name, "split": "validation",
#                       "wcc_surrogate_l_d": yva,
#                       "wcc_ann_l_d": net.predict(Xva_s)}),
#         pd.DataFrame({"livestock_type": name, "split": "test",
#                       "wcc_surrogate_l_d": yte,
#                       "wcc_ann_l_d": pred}),
#     ], ignore_index=True)

#     # Ceiling: the best any climate-only model could do, given that
#     # physiology moves the target at fixed climate. Estimated by binning on
#     # temperature and taking the within-bin mean as the best possible
#     # prediction. It bins on one variable only, so it understates what a
#     # four-variable model can reach -- treat it as a floor on the ceiling.
#     bins = pd.qcut(pd.Series(Xte[:, 0]), 40, duplicates="drop")
#     within = pd.Series(yte).groupby(bins, observed=True).transform("mean")
#     ceiling = r2_score(yte, within)

#     stats = {
#         "livestock_type": name,
#         "n_samples": len(samples),
#         "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
#         "r2_test": r2_score(yte, pred),
#         "rmse_test_l_d": float(np.sqrt(mean_squared_error(yte, pred))),
#         "r2_ceiling_climate_only": ceiling,
#         "pct_of_ceiling": float(100 * r2_score(yte, pred) / ceiling)
#         if ceiling > 0 else np.nan,
#         "features": ",".join(feats),
#         "hidden_layers": str(hidden),
#         "epochs_run": len(history),
#         "best_epoch": best_epoch,
#         "best_val_rmse_l_d": best_val,
#     }

#     rep(f"\n  {name}")
#     rep(f"    ANN {hidden}, {stats['epochs_run']} epochs, best at "
#         f"{best_epoch} (val RMSE {best_val:.4f} L/d)")
#     rep(f"    R2 test {stats['r2_test']:.4f} | climate-only ceiling "
#         f"{ceiling:.4f} ({stats['pct_of_ceiling']:.1f}% of it reached)")
#     rep(f"    RMSE test {stats['rmse_test_l_d']:.4f} L/d")
#     rep("    the ceiling is below 1 by construction: at one climate the "
#         "target")
#     rep("    takes many values because physiology varied, and no "
#         "climate-only")
#     rep("    input can separate them.")
#     return net, scaler, stats, pd.DataFrame(history), preds


# # ---------------------------------------------------------------------------
# # county application
# # ---------------------------------------------------------------------------

# def load_county_climate(climate_dir: Path, rep: Report) -> pd.DataFrame:
#     """
#     Assemble the county climate predictors from the per-forcing tables.
#     Every table shares the same keys, so they join cleanly on fips.
#     """
#     stems = {"Temp": "temp_c_county", "RH": "rh_pct_county",
#              "Wind": "wind2_kmh_county", "Sunlight": "sunlight_h_county"}
#     out = None
#     for stem, col in stems.items():
#         hits = sorted(climate_dir.glob(f"{stem}_county_level_*.feather"))
#         if not hits:
#             raise FileNotFoundError(
#                 f"no {stem}_county_level_*.feather in {climate_dir}. "
#                 f"Run era5_forcings.py first.")
#         df = pd.read_feather(hits[-1])
#         keys = [k for k in ("year", "month", "fips", "state_fips")
#                 if k in df.columns]
#         sub = df[keys + [col]].rename(
#             columns={col: col.replace("_county", "")})
#         out = sub if out is None else out.merge(sub, on=keys, how="inner")

#     out = out.rename(columns=ERA5_RENAME)
#     out["fips"] = out["fips"].astype(str).str.zfill(5)
#     rep(f"  county climate: {len(out):,} rows, "
#         f"{out['fips'].nunique():,} counties, "
#         f"{out['year'].min()}-{out['year'].max()}")

#     # Missing values are carried through as NaN rather than dropped or
#     # imputed, so a county with no climate ends up with no WCC instead of a
#     # fabricated one. Reported per variable because the cause differs:
#     # a county too small to hold an ERA5 pixel centre loses every variable,
#     # whereas sunlight_h alone goes missing where the radiation ratio
#     # cannot be inverted.
#     miss = {c: int(out[c].isna().sum()) for c in CLIMATE_FEATURES
#             if c in out.columns}
#     bad = {k: v for k, v in miss.items() if v}
#     if bad:
#         rep(f"  missing values by variable: {bad}")
#         n_rows = int(out[CLIMATE_FEATURES].isna().any(axis=1).sum())
#         n_cty = int(out.loc[out[CLIMATE_FEATURES].isna().any(axis=1),
#                             "fips"].nunique())
#         rep(f"  -> {n_rows:,} rows ({100*n_rows/len(out):.2f}%) across "
#             f"{n_cty:,} counties will receive no WCC")
#     else:
#         rep("  no missing climate values")
#     return out


# def load_composition(usda_dir: Path, name: str, years: np.ndarray,
#                      rep: Report) -> Optional[pd.DataFrame]:
#     """
#     County herd composition, interpolated from the census years onto every
#     climate year.

#     The census reports 2002, 2007, 2012, 2017 and 2022. A composition
#     FRACTION changes far more slowly and smoothly than a head count, so
#     linear interpolation between census years is defensible where a
#     head-count interpolation would not be. Outside the census span the
#     nearest observed value is held constant, and every row is labelled with
#     how it was obtained so an extrapolated value is never mistaken for an
#     observed one.
#     """
#     pattern = COMPOSITION_SOURCE.get(name)
#     if pattern is None:
#         return None
#     hits = sorted(usda_dir.glob(pattern))
#     if not hits:
#         rep(f"  {name}: no {pattern} in {usda_dir}; composition unavailable")
#         return None

#     col = [c for c in COMPOSITION_FEATURES[name]][0]
#     target = COMPOSITION_FEATURES[name][col]
#     df = pd.read_feather(hits[-1])
#     if col not in df.columns:
#         rep(f"  {name}: {hits[-1].name} has no {col}; composition unavailable")
#         return None

#     df = df[["year", "fips", col]].dropna()
#     df["fips"] = df["fips"].astype(str).str.zfill(5)
#     census_years = sorted(df["year"].unique())

#     out = []
#     for fips, g in df.groupby("fips"):
#         g = g.sort_values("year")
#         vals = np.interp(years, g["year"].to_numpy(dtype=float),
#                          g[col].to_numpy(dtype=float))
#         origin = np.where(
#             years < g["year"].min(), "held from earliest census",
#             np.where(years > g["year"].max(), "held from latest census",
#                      "interpolated between census years"))
#         out.append(pd.DataFrame({"fips": fips, "year": years,
#                                  target: vals,
#                                  f"{target}_origin": origin}))
#     res = pd.concat(out, ignore_index=True)

#     n_in = int((res[f"{target}_origin"] == "interpolated between census years").sum())
#     rep(f"  {name}: {col} for {res['fips'].nunique():,} counties, census "
#         f"years {[int(y) for y in census_years]}")
#     rep(f"    {n_in:,} of {len(res):,} county-years interpolated within the "
#         f"census span; the rest held from the nearest census year")
#     return res


# def coverage_check(rep: Report, name: str, samples: pd.DataFrame,
#                    county: pd.DataFrame) -> dict:
#     """
#     How much county climate falls outside the range the ANN was trained on.

#     A neural network extrapolates arbitrarily, without warning, so this is
#     the number that decides whether a county prediction can be trusted.
#     """
#     rep(f"\n  {name}: county climate against the training range")
#     frac_out = {}
#     for f in county_features(name):
#         if f not in county.columns:
#             rep(f"    {f:<12} NOT PRESENT in the county table")
#             frac_out[f] = 1.0
#             continue
#         lo, hi = samples[f].min(), samples[f].max()
#         v = county[f].dropna()
#         if v.empty:
#             rep(f"    {f:<12} no finite county values")
#             frac_out[f] = 1.0
#             continue
#         out = float(((v < lo) | (v > hi)).mean())
#         frac_out[f] = out
#         flag = "   <-- EXTRAPOLATING" if out > 0.01 else ""
#         rep(f"    {f:<12} train [{lo:8.2f}, {hi:8.2f}]  county "
#             f"[{v.min():8.2f}, {v.max():8.2f}]  outside {100*out:5.2f}%{flag}")
#     return frac_out


# def complete_mask(county: pd.DataFrame, feats: List[str]) -> np.ndarray:
#     """Rows where every predictor the model needs is finite."""
#     have = [f for f in feats if f in county.columns]
#     if len(have) < len(feats):
#         return np.zeros(len(county), dtype=bool)
#     return np.isfinite(county[have].to_numpy(dtype=float)).all(axis=1)


# def apply_to_counties(net, scaler, county: pd.DataFrame,
#                       feats: List[str]) -> np.ndarray:
#     """
#     Predict on complete rows only.

#     A neural network cannot take NaN, and imputing a climate value would
#     invent a county's weather. Incomplete rows stay NaN in the output so
#     that a missing county is visibly missing downstream.
#     """
#     out = np.full(len(county), np.nan)
#     ok = complete_mask(county, feats)
#     if ok.any():
#         X = county.loc[ok, feats].to_numpy(dtype=float)
#         out[ok] = net.predict(scaler.transform(X))
#     return out


# # ---------------------------------------------------------------------------
# # driver
# # ---------------------------------------------------------------------------

# def run(data_dir: str = DATA_DIR, wcc_dir: str = WCC_DIR,
#         climate_dir: str = CLIMATE_DIR, usda_dir: str = USDA_DIR,
#         n_samples: int = N_SAMPLES,
#         seed: int = SEED, hidden: Tuple[int, ...] = (64, 32),
#         max_iter: int = 400, broiler_fraction: float = 0.78,
#         patience: int = 25, verbose: bool = False
#         ) -> Dict[str, pd.DataFrame]:
#     base = Path(data_dir).expanduser()
#     base.mkdir(parents=True, exist_ok=True)
#     rep = Report(echo=verbose)

#     # The MLR module supplies both the samplers and the fitted equations,
#     # so the surrogate targets here are the same construction that produced
#     # wcc_mlr_coefficients.feather.
#     sys.path.insert(0, str(Path(__file__).resolve().parent))
#     import importlib
#     wcc_mod = importlib.import_module("wcc_mlr")

#     coef_path = Path(wcc_dir).expanduser() / "wcc_mlr_coefficients.feather"
#     if not coef_path.exists():
#         raise FileNotFoundError(
#             f"{coef_path} not found. Run wcc_mlr.py first.")
#     coef_df = pd.read_feather(coef_path)

#     county_climate = load_county_climate(Path(climate_dir).expanduser(), rep)
#     usda_path = Path(usda_dir).expanduser()
#     all_years = np.sort(county_climate["year"].unique()).astype(float)
#     monthly = "month" in county_climate.columns
#     y0, y1 = int(county_climate["year"].min()), int(county_climate["year"].max())

#     outputs: Dict[str, pd.DataFrame] = {}
#     all_stats, histories, predictions = [], [], []

#     bar = tqdm(LIVESTOCK, desc="Downscaling WCC", unit="type")
#     for name in bar:
#         bar.set_postfix_str(name)
#         row = coef_df[coef_df["livestock_type"] == name]
#         if row.empty:
#             rep(f"  {name}: no fitted equation, skipped")
#             continue
#         coefs = row.iloc[0].to_dict()

#         # Attach observed county composition where the census reports it.
#         comp = load_composition(usda_path, name, all_years, rep)
#         county_in = county_climate
#         if comp is not None:
#             county_in = county_climate.merge(
#                 comp, on=["fips", "year"], how="left")

#         samples = make_surrogate(wcc_mod, name, coefs, n_samples, seed,
#                                  broiler_fraction)
#         net, scaler, stats, hist, preds = train_ann(
#             rep, name, samples, hidden, seed, max_iter, patience)
#         histories.append(hist)
#         predictions.append(preds)
#         cov = coverage_check(rep, name, samples, county_in)
#         stats.update({f"frac_outside_{k}": v for k, v in cov.items()})
#         all_stats.append(stats)

#         feats = county_features(name)
#         out = county_in[[c for c in
#                          ("year", "month", "fips", "state_fips")
#                          if c in county_in.columns]].copy()
#         out["wcc_ann_l_d"] = apply_to_counties(net, scaler, county_in, feats)
#         out["wcc_analytic_l_d"] = analytic_transfer(
#             wcc_mod, name, coefs, samples, county_in)
#         for extra in COMPOSITION_FEATURES[name].values():
#             if extra in county_in:
#                 out[extra] = county_in[extra].to_numpy()
#                 oc = f"{extra}_origin"
#                 if oc in county_in:
#                     out[oc] = county_in[oc].to_numpy()
#         out["wcc_ann_gal_d"] = out["wcc_ann_l_d"] * GAL_PER_L
#         out["livestock_type"] = name

#         both = out[["wcc_ann_l_d", "wcc_analytic_l_d"]].dropna()
#         rep(f"\n  {name}: ANN against the closed-form transfer")
#         if len(both) > 1:
#             d = both["wcc_ann_l_d"] - both["wcc_analytic_l_d"]
#             denom = both["wcc_analytic_l_d"].abs().replace(0, np.nan)
#             rep(f"    mean abs difference {d.abs().mean():.4f} L/d "
#                 f"({100 * (d.abs() / denom).mean():.2f}% of the analytic value)")
#             rep(f"    correlation "
#                 f"{np.corrcoef(both['wcc_ann_l_d'], both['wcc_analytic_l_d'])[0,1]:.6f}")
#             all_stats[-1]["ann_vs_analytic_mae_l_d"] = float(d.abs().mean())
#         else:
#             rep("    not computable: no rows with complete climate")
#             all_stats[-1]["ann_vs_analytic_mae_l_d"] = np.nan

#         n_nan = int(out["wcc_ann_l_d"].isna().sum())
#         if n_nan:
#             rep(f"    {n_nan:,} rows ({100*n_nan/len(out):.2f}%) left as NaN "
#                 f"for want of complete county climate")
#         all_stats[-1]["n_county_rows_no_climate"] = n_nan

#         n_neg = int((out["wcc_ann_l_d"] < 0).sum())
#         if n_neg:
#             rep(f"    NOTE: {n_neg:,} negative predictions "
#                 f"({100*n_neg/len(out):.3f}%), reported not clipped")

#         outputs[f"wcc_county_{name}_{y0}_{y1}.feather"] = out

#         if monthly:
#             keys = ["year", "fips", "state_fips"]
#             # mean skips NaN months; a county-year with no valid month
#             # stays NaN rather than becoming zero.
#             ann = (out.groupby(keys, as_index=False)
#                    .agg(wcc_ann_l_d=("wcc_ann_l_d", "mean"),
#                         wcc_ann_gal_d=("wcc_ann_gal_d", "mean"),
#                         wcc_analytic_l_d=("wcc_analytic_l_d", "mean"),
#                         n_months=("wcc_ann_l_d", "count")))
#             ann["livestock_type"] = name
#             outputs[f"wcc_county_{name}_annual_{y0}_{y1}.feather"] = ann
#     bar.close()

#     outputs["wcc_ann_performance.feather"] = pd.DataFrame(all_stats)
#     if histories:
#         # Per-epoch curves, so the training run can be plotted afterwards
#         # rather than only summarised.
#         outputs["wcc_ann_training_history.feather"] = pd.concat(
#             histories, ignore_index=True)
#     if predictions:
#         outputs["wcc_ann_predictions.feather"] = pd.concat(
#             predictions, ignore_index=True)

#     for fname, df in outputs.items():
#         df.reset_index(drop=True).to_feather(base / fname)

#     if not verbose:
#         print("Done.")
#         for s in all_stats:
#             print(f"  {s['livestock_type']:<14}R2 {s['r2_test']:.4f}  "
#                   f"(ceiling {s['r2_ceiling_climate_only']:.4f}, "
#                   f"{s['pct_of_ceiling']:.0f}% reached)  "
#                   f"{s['epochs_run']:>3} epochs, best {s['best_epoch']:>3}  "
#                   f"vs closed form "
#                   f"{s.get('ann_vs_analytic_mae_l_d', float('nan')):.4f} L/d")
#         print(f"  granularity  {'county-month' if monthly else 'county-year'}")
#         print(f"  files        {len(outputs)} written")
#         print(f"  tables       {base}")
#     return outputs


# def main(argv: Optional[List[str]] = None) -> int:
#     ap = argparse.ArgumentParser(
#         description="Transfer generic MLR WCC to county level with an ANN.",
#         allow_abbrev=False)
#     ap.add_argument("--data-dir", default=DATA_DIR)
#     ap.add_argument("--wcc-dir", default=WCC_DIR)
#     ap.add_argument("--climate-dir", default=CLIMATE_DIR)
#     ap.add_argument("--usda-dir", default=USDA_DIR,
#                     help="census tables holding the herd composition")
#     ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
#     ap.add_argument("--seed", type=int, default=SEED)
#     ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
#     ap.add_argument("--max-iter", type=int, default=400,
#                     help="maximum training epochs per livestock type")
#     ap.add_argument("--patience", type=int, default=25,
#                     help="epochs without validation improvement before "
#                          "stopping")
#     ap.add_argument("--broiler-fraction", type=float, default=0.78)
#     ap.add_argument("--verbose", action="store_true")
#     args, unknown = ap.parse_known_args(argv)
#     if unknown:
#         print(f"(ignoring unrecognized arguments: {unknown})")
#     run(args.data_dir, args.wcc_dir, args.climate_dir, args.usda_dir,
#         args.n_samples, args.seed, tuple(args.hidden), args.max_iter,
#         args.broiler_fraction, args.patience, args.verbose)
#     return 0


# if __name__ == "__main__":
#     main()