"""
wcc_mlr.py

Water consumption coefficients (WCC) for dairy cattle, beef cattle, hogs and
poultry: literature-based sample generation, then a multiple linear
regression surrogate fitted per livestock type.

    <DATA_DIR>/
        wcc_samples_dairy_cattle.feather
        wcc_samples_beef_cattle.feather
        wcc_samples_hogs.feather
        wcc_samples_poultry.feather
        wcc_mlr_coefficients.feather
        wcc_mlr_equations.csv

    <META_DIR>/ wcc_qa.txt

--------------------------------------------------------------------------
WHAT THIS DOES, AND WHAT IT DOES NOT CLAIM
--------------------------------------------------------------------------
Samples are drawn over physiological and climatic states, and the WCC for
each sample is computed from a PUBLISHED water intake equation. The MLR is
then fitted to those samples.

The MLR is therefore a LINEAR SURROGATE of published intake equations, not
an independent empirical model fitted to field observations. It can only
recover what the source equations encode. Describing it as anything more
would misstate the method.

Deterministic equations would let the MLR recover them almost exactly, so a
multiplicative residual is added to every sample, with a coefficient of
variation taken from the prediction error published for each equation
(e.g. RMSPE 14.4 percent for the dairy equation of Appuhamy et al., 2016).
Without it the reported R-squared would be an artifact of the generator.

Inputs are sampled inside each equation's stated validity domain, so
outputs land in a plausible range without hard clamping. Values falling
outside published plausibility bounds are COUNTED AND REPORTED, never
pinned to a bound: clamping piles probability mass on the boundary and
destroys the variance the regression needs.

--------------------------------------------------------------------------
SOURCE EQUATIONS
--------------------------------------------------------------------------
DAIRY -- Murphy, M.R., C.L. Davis and G.C. McCoy (1983), J. Dairy Sci.
66:35-38; adopted by NRC (2001).

    FWI (kg/d) = 15.99 + 1.58*DMI + 0.90*MY + 0.05*NaI + 1.20*Tmin

  DMI kg/d, MY milk yield kg/d, NaI sodium intake g/d, Tmin daily minimum
  air temperature degC. Chosen as primary because milk yield enters
  explicitly, so lactating and dry cows are handled by one equation, and
  lactation status is a required feature of the framework.

  Cross-checked against NASEM (2021) Eq 9-1, after Appuhamy et al. (2016),
  J. Dairy Sci. 99:7191-7205:

    FWI (L/d) = -91.1 + 2.93*DMI + 0.61*DM% + 0.062*NaK + 2.49*CP% + 0.76*T

BEEF -- NASEM (2016), Nutrient Requirements of Beef Cattle, 8th rev. ed.,
Eq 19-105 (R-squared = 0.997), with the Current Effective Temperature Index
of Eq 19-103:

    WI (L/d) = 7.3 + 0.0805*SBW - 0.00008*SBW^2
               - 1.225*CETI + 0.0411*CETI^2 + 0.0023268*SBW*CETI

    CETI = 27.88 - 0.456*Tc + 0.010754*Tc^2 - 0.4905*RH + 0.00088*RH^2
           + 1.1507*ws - 0.126447*ws^2 + 0.019867*Tc*RH
           - 0.046313*Tc*ws + 0.41267*HRS,     ws = WS/3.6

  SBW shrunk body weight = 0.96*BW. Tc degC, RH percent, WS wind speed
  km/h, HRS hours of sunlight. Verified against the worked example in NMSU
  Guide B-231: Tc 26.7, RH 30, WS 10, HRS 12 gives CETI 29.10 (published
  29.1) and, at SBW 479, WI 59.09 L/d (published 59.1). Both are asserted
  at import; the module refuses to run if they do not reproduce.

  This equation is the reason beef carries humidity, wind and sunlight as
  features: CETI is built from them, and all four are available from
  ERA5-Land for the downstream county work.

  Cross-checked against Winchester, C.F. and M.J. Morris (1956), J. Anim.
  Sci. 15:722-740:   WI = DMI * (3.413 + 0.01595 * exp(0.17596*T))

HOGS -- no single canonical equation exists. Water intake is built from the
water-to-feed ratio, which is how the swine literature reports it: 2:1 to
3:1 for nursery and grow-finish pigs, declining as pigs grow (Shaw et al.,
2006, via Kansas State University Swine Nutrition Guide), rising under heat
load. Class-level daily intakes from extension guidance anchor the result:
nursery 0.5-1.0, grower 2-3, finisher 3-5, gestating and lactating sows
5-7 gal/d.

POULTRY -- a mixture of broilers and layers, because the census category
combines them.

  Broilers: Pesti, G.M., S.V. Amato and L.R. Minear (1985), Poult. Sci.
  64:803-808. Water consumption is a linear function of age,
  R-squared > 0.99:

    WI (mL/d) = c(T) * age_days,    c = 5.28 overall,
                                    5.1 in cooler months,
                                    5.7 in the warmest months

  Layers: water-to-feed ratio against temperature, from Lohmann Breeders
  layer management guidance and industry heat-stress advice: 1.82:1 at
  15 degC, about 2:1 at 21 degC, 4.9:1 at 30-35 degC, up to 8:1 at 38 degC.

Usage
-----
    python wcc_mlr.py
    python wcc_mlr.py --n-samples 100000 --seed 7
    from wcc_mlr import run; samples, coefs = run()
"""

from __future__ import annotations

import argparse
import datetime as dt
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
DATA_DIR = f"{_REPO}/data/wcc_mlr"
META_DIR = f"{_REPO}/data/_meta/wcc"

N_SAMPLES = 50_000          # manuscript used 10,000
SEED = 42
L_PER_GAL = 3.785411784
GAL_PER_L = 1.0 / L_PER_GAL

# Residual CV per type, from the prediction error published with each
# equation. Dairy: RMSPE 14.4% (Appuhamy et al., 2016). Beef: the equation
# is reported at R2 = 0.997 in sample, but individual animal variation is
# far larger, so the CV is set from observed between-animal spread rather
# than the fit statistic. Poultry: Pesti reports R2 > 0.99 across flocks.
# Between-HERD variation in a county-year average, not between-animal
# variation. The unit of analysis is one county-year herd or flock average,
# because that is what a WCC multiplies. Averaging over thousands of head
# damps individual scatter, so this is far smaller than the published
# per-animal prediction errors (RMSPE 14.4 percent for the dairy equation of
# Appuhamy et al., 2016). What remains is genuine between-herd spread in
# management, genetics and housing that the features do not carry.
RESIDUAL_CV = {"dairy_cattle": 0.075, "beef_cattle": 0.070,
               "hogs": 0.080, "poultry": 0.070}

# Service water: everything delivered to the animals beyond drinking --
# parlour and pen washing, evaporative cooling, drinker spillage, waste
# flushing. Published intake equations give DRINKING water only, whereas
# livestock water USE per head is larger. Set --basis total to apply these,
# or leave the default drinking basis and they are all 1.0.
#
# THESE FACTORS ARE AN ASSUMPTION AND ARE NOT TAKEN FROM A SINGLE SOURCE.
# They must be replaced with cited values before publication; they are
# exposed here so the choice is visible rather than buried in a fudge.
SERVICE_WATER_FACTOR = {"dairy_cattle": 1.50, "beef_cattle": 1.90,
                        "hogs": 3.90, "poultry": 3.10}

# Plausibility bounds used ONLY for reporting. Nothing is clamped to them.
PLAUSIBLE_L_PER_DAY = {
    "dairy_cattle": (25.0, 250.0),
    "beef_cattle": (10.0, 120.0),
    "hogs": (1.0, 45.0),
    "poultry": (0.01, 1.20),
}

# Share of the national poultry inventory that is broilers rather than
# layers. The census category combines them, so one WCC must represent the
# mixture. Override with --broiler-fraction to match the head counts in
# usda_livestock_county_2002_2022.feather.
BROILER_FRACTION = 0.78

# How much of the tabulated physiological spread to keep. The literature
# describes individual animals; a county WCC is a herd average, and herd
# averages differ from one another far less than animals do. 1.0 keeps the
# between-animal spread, 0.25 is the between-county default.
#
# It is not a tuning knob for making plots look better: it changes what one
# sample REPRESENTS. At 1.0 a row is an animal, and climate explains 65
# percent of dairy water use and 8 percent of hog water use. At 0.25 a row
# is a county herd, and the figures are 95 and 50 percent. The second is
# the quantity the pipeline actually needs.
HERD_SPREAD = 0.25

# ---------------------------------------------------------------------------
# LITERATURE-COMPILED PHYSIOLOGY AND WCC (manuscript Table 1)
# ---------------------------------------------------------------------------
# Compiled from the cited studies, technical documents and reports. These
# statistics ARE the calibration target: the sampler reproduces them, and
# the WCC of every sample is mapped onto the WCC distribution below, so the
# central tendency the literature reports survives into the MLR fit.
#
# TWO CORRECTIONS APPLIED TO THE TABLE AS SUPPLIED:
#
# 1. Body weight is stored here in KILOGRAMS. The source table heads that
#    column "lbs", but read as pounds a dairy cow weighs 266 kg and a pig
#    33 kg, which contradicts the same table's own maximum ages of 60 and
#    23 months. Read as kilograms every row is physiologically sensible.
#    The header is wrong, not the data, and it should be fixed in the
#    manuscript.
#
# 2. The poultry WCC row is internally impossible as supplied: mean 0.04
#    gal/d sits BELOW its own 25th percentile of 0.08, with a minimum of
#    0.01. No distribution can satisfy that. The PERCENTILES are taken as
#    authoritative here and the mean is treated as a typo, which implies a
#    mean near 0.10 gal/d. Change POULTRY_WCC_USE_MEAN below to flip that
#    decision.
#
# WCC is stored in gal/d/animal exactly as tabulated; the sampler converts.
POULTRY_WCC_USE_MEAN = False

LITERATURE = {
    "dairy_cattle": {
        "refs": "12,15,17,18,20-24",
        "age_months": dict(mean=30, std=17, min=1, q25=15, q50=25, q75=45, max=60),
        "bw_kg":      dict(mean=587.03, std=79.51, min=454.86, q25=517.82,
                           q50=588.81, q75=656.17, max=725.68),
        "lm_l_d":     dict(mean=10.14, std=5.85, min=0.07, q25=4.82,
                           q50=10.38, q75=15.21, max=19.99),
        "dmi_pct":    dict(mean=5.97, std=2.41, min=2.12, q25=3.75,
                           q50=5.56, q75=8.21, max=10.88),
        "wcc_gal_d":  dict(mean=33.4, std=12.1, min=15.25, q25=20.5,
                           q50=36.4, q75=37.3, max=70.15),
    },
    "beef_cattle": {
        "refs": "15,16,18,21,25-27,28",
        "age_months": dict(mean=26, std=14, min=1, q25=13, q50=23, q75=32, max=58),
        "bw_kg":      dict(mean=616.38, std=83.48, min=477.60, q25=543.71,
                           q50=618.25, q75=688.98, max=761.97),
        "lm_l_d":     None,
        "dmi_pct":    dict(mean=8.96, std=3.62, min=3.18, q25=5.62,
                           q50=8.35, q75=12.32, max=16.33),
        "wcc_gal_d":  dict(mean=12.7, std=7.57, min=4.68, q25=8.9,
                           q50=12.5, q75=12.8, max=22.35),
    },
    "hogs": {
        "refs": "15,18,26,31,32",
        "age_months": dict(mean=10, std=7, min=1, q25=5, q50=8, q75=15, max=23),
        "bw_kg":      dict(mean=73.22, std=37.89, min=2.27, q25=41.35,
                           q50=74.70, q75=105.83, max=135.87),
        "lm_l_d":     dict(mean=6.67, std=6.65, min=0.00, q25=0.00,
                           q50=4.98, q75=12.47, max=20.00),
        "dmi_pct":    dict(mean=3.81, std=2.27, min=0.07, q25=1.57,
                           q50=3.95, q75=5.68, max=8.24),
        "wcc_gal_d":  dict(mean=4.69, std=2.98, min=1.28, q25=3.06,
                           q50=4.18, q75=5.35, max=15.07),
    },
    "poultry": {
        "refs": "15,18,33,34",
        "age_months": dict(mean=14, std=8, min=1, q25=7, q50=11, q75=20, max=28),
        "bw_kg":      dict(mean=1.88, std=0.97, min=0.20, q25=1.08,
                           q50=1.85, q75=2.71, max=3.50),
        "lm_l_d":     None,
        "dmi_pct":    dict(mean=0.09, std=0.05, min=0.01, q25=0.05,
                           q50=0.09, q75=0.13, max=0.29),
        "wcc_gal_d":  dict(mean=0.04, std=0.06, min=0.01, q25=0.08,
                           q50=0.10, q75=0.13, max=0.25),
    },
}

# Rank correlation imposed between the physiological variables. Sampling
# them independently would pair a one-month-old animal with a mature body
# weight; the table gives marginals only, so the dependence has to be
# supplied. Values are the obvious physiological orderings -- age drives
# weight, weight drives intake -- not fitted quantities.
PHYS_CORR = {
    ("age_months", "bw_kg"): 0.80,
    ("age_months", "lm_l_d"): 0.35,
    ("age_months", "dmi_pct"): 0.25,
    ("bw_kg", "lm_l_d"): 0.45,
    ("bw_kg", "dmi_pct"): 0.30,
    ("lm_l_d", "dmi_pct"): 0.40,
}



LIVESTOCK = ("dairy_cattle", "beef_cattle", "hogs", "poultry")

# Physiological predictors only, matching the published Table 2 form:
# age (A), body weight (BW), dry matter intake (DMI) and the lactating,
# breeding or layer share (L). No climatic, interaction or higher-order
# terms, so each fitted model is a single-line equation.
# Lactation (L) applies only where the animal lactates: dairy cattle and
# lactating sows. Beef cattle and poultry take age, body weight and intake
# alone. The poultry layer share is a flock-composition attribute, not a
# lactation term, and is not used as one.
BASE_FEATURES = {
    "dairy_cattle": ["age_months", "bw_kg", "dmi_kg_d", "lactating"],
    "beef_cattle": ["age_months", "bw_kg", "dmi_kg_d"],
    "hogs": ["age_months", "bw_kg", "dmi_kg_d", "lactating_sows"],
    "poultry": ["age_months", "bw_kg", "dmi_kg_d"],
}

INTERACTIONS = {k: {} for k in BASE_FEATURES}

FEATURES = {k: list(BASE_FEATURES[k]) for k in BASE_FEATURES}


def apply_basis(df: pd.DataFrame, name: str, basis: str) -> pd.DataFrame:
    """
    Scale from drinking water to total on-farm livestock water if asked.
    SERVICE_WATER_FACTOR is an assumption pending cited values, and is
    applied only when basis="total" so the choice is never silent.
    """
    if basis != "total":
        return df
    k = SERVICE_WATER_FACTOR[name]
    for col in ("wcc_expected_l_d", "wcc_expected_gal_d",
                "wcc_l_d", "wcc_gal_d", "crosscheck_l_d"):
        if col in df:
            df[col] = df[col] * k
    return df


def add_interactions(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Materialise the interaction columns for one livestock type."""
    df = df.copy()
    for col, fn in INTERACTIONS[name].items():
        df[col] = fn(df)
    return df


class Report:
    """
    Diagnostics sink. With path=None nothing is written to disk; the
    narrative is discarded unless echo=True sends it to the console.
    """

    def __init__(self, path: Optional[Path] = None, echo: bool = False):
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


# ---------------------------------------------------------------------------
# published equations
# ---------------------------------------------------------------------------

def ceti(tc, rh, ws_kmh, hrs):
    """NASEM (2016) Eq 19-103, Current Effective Temperature Index."""
    ws = np.asarray(ws_kmh) / 3.6                       # km/h -> m/s
    tc = np.asarray(tc, dtype=float)
    rh = np.asarray(rh, dtype=float)
    hrs = np.asarray(hrs, dtype=float)
    return (27.88 - 0.456 * tc + 0.010754 * tc ** 2
            - 0.4905 * rh + 0.00088 * rh ** 2
            + 1.1507 * ws - 0.126447 * ws ** 2
            + 0.019867 * tc * rh - 0.046313 * tc * ws
            + 0.41267 * hrs)


def beef_wi_nasem(bw_kg, tc, rh, ws_kmh, hrs):
    """NASEM (2016) Eq 19-105, water intake L/d. SBW = 0.96 * BW."""
    sbw = np.asarray(bw_kg, dtype=float) * 0.96
    ct = ceti(tc, rh, ws_kmh, hrs)
    return (7.3 + 0.0805 * sbw - 0.00008 * sbw ** 2
            - 1.225 * ct + 0.0411 * ct ** 2 + 0.0023268 * sbw * ct)


def beef_wi_winchester(dmi_kg_d, tc):
    """Winchester and Morris (1956), cross-check only."""
    dmi = np.asarray(dmi_kg_d, dtype=float)
    tc = np.asarray(tc, dtype=float)
    return dmi * (3.413 + 0.01595 * np.exp(0.17596 * tc))


def dairy_fwi_murphy(dmi_kg_d, milk_kg_d, na_g_d, tmin_c):
    """Murphy et al. (1983), free water intake kg/d, adopted by NRC (2001)."""
    return (15.99 + 1.58 * np.asarray(dmi_kg_d, dtype=float)
            + 0.90 * np.asarray(milk_kg_d, dtype=float)
            + 0.05 * np.asarray(na_g_d, dtype=float)
            + 1.20 * np.asarray(tmin_c, dtype=float))


def dairy_fwi_nasem(dmi_kg_d, dm_pct, nak, cp_pct, tmean_c):
    """NASEM (2021) Eq 9-1, after Appuhamy et al. (2016). Cross-check."""
    return (-91.1 + 2.93 * np.asarray(dmi_kg_d, dtype=float)
            + 0.61 * np.asarray(dm_pct, dtype=float)
            + 0.062 * np.asarray(nak, dtype=float)
            + 2.49 * np.asarray(cp_pct, dtype=float)
            + 0.76 * np.asarray(tmean_c, dtype=float))


def swine_water_feed_ratio(tc):
    """
    Water-to-feed ratio against temperature. Anchored on the swine
    literature: 2:1 to 3:1 for nursery and grow-finish pigs at
    thermoneutral conditions, rising under heat load.
    """
    tc = np.asarray(tc, dtype=float)
    return np.interp(tc, [0.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
                     [2.0, 2.1, 2.4, 2.9, 3.5, 4.2, 4.8])


def layer_water_feed_ratio(tc):
    """
    Layer water-to-feed ratio against temperature: 1.82:1 at 15 degC,
    about 2:1 at 21 degC, 4.9:1 at 30-35 degC, up to 8:1 at 38 degC.
    """
    tc = np.asarray(tc, dtype=float)
    return np.interp(tc, [0.0, 15.0, 21.0, 27.0, 32.5, 38.0, 45.0],
                     [1.70, 1.82, 2.00, 3.10, 4.90, 8.00, 9.00])


def broiler_coefficient(tc):
    """
    Pesti et al. (1985): WI (mL/d) = c * age_days, with c = 5.28 overall,
    5.1 in cooler months and 5.7 in the warmest. Interpolated on
    temperature and extended for heat load above 30 degC.
    """
    tc = np.asarray(tc, dtype=float)
    return np.interp(tc, [0.0, 10.0, 21.0, 27.0, 32.0, 38.0, 45.0],
                     [4.70, 5.10, 5.28, 5.70, 6.60, 8.50, 9.50])


# Refuse to run if the published worked example does not reproduce.
_ceti_chk = float(ceti(26.7, 30.0, 10.0, 12.0))
_wi_chk = float(beef_wi_nasem(479.0 / 0.96, 26.7, 30.0, 10.0, 12.0))
assert abs(_ceti_chk - 29.1) < 0.05, f"CETI check failed: {_ceti_chk}"
assert abs(_wi_chk - 59.1) < 0.1, f"NASEM beef check failed: {_wi_chk}"


# ---------------------------------------------------------------------------
# sample generation
# ---------------------------------------------------------------------------

def quantile_function(q: dict):
    """
    Monotone inverse CDF through the five tabulated quantiles.

    PCHIP rather than a spline: it is shape-preserving, so it cannot
    overshoot below the tabulated minimum or above the maximum, which a
    cubic spline through the same five points can do.
    """
    from scipy.interpolate import PchipInterpolator
    probs = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    vals = np.array([q["min"], q["q25"], q["q50"], q["q75"], q["max"]],
                    dtype=float)
    vals = np.maximum.accumulate(vals)          # guard a non-monotone row
    return PchipInterpolator(probs, vals, extrapolate=False)


def sample_physiology(rng, n: int, name: str) -> pd.DataFrame:
    """
    Draw age, body weight, milk yield and intake so that each marginal
    reproduces the tabulated quantiles, with a physiologically sensible
    dependence between them.

    A Gaussian copula supplies the dependence: correlated normals are
    turned into uniforms, then each uniform is pushed through its own
    variable's quantile function. The marginals are therefore exactly the
    literature's, while age still tracks body weight instead of pairing a
    one-month-old with a mature carcass.
    """
    spec = LITERATURE[name]
    cols = [c for c in ("age_months", "bw_kg", "lm_l_d", "dmi_pct")
            if spec.get(c) is not None]

    k = len(cols)
    corr = np.eye(k)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:
                r = PHYS_CORR.get((a, b), PHYS_CORR.get((b, a), 0.0))
                corr[i, j] = corr[j, i] = r

    # Nearest positive-definite correction, in case the stated
    # correlations are not jointly consistent.
    w, V = np.linalg.eigh(corr)
    corr = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    z = rng.multivariate_normal(np.zeros(k), corr, size=n)
    from scipy.stats import norm
    u = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)

    out = pd.DataFrame(index=range(n))
    for i, c in enumerate(cols):
        out[c] = quantile_function(spec[c])(u[:, i])

    # THE DMI COLUMN IS kg/day, NOT A PERCENTAGE OF BODY WEIGHT, despite
    # the header. Read as a percentage it gives a finishing steer 51.6 kg
    # of dry matter a day against a real 8-12, and a laying hen 0.0002 kg
    # against a real 0.11 -- impossible in both directions. Read as kg/day
    # the beef (8.35), hog (3.95) and poultry (0.09) medians all land on
    # their published intakes. This is the same class of error as the body
    # weight column, whose header says lbs but whose values are only
    # sensible as kg; both should be corrected in the manuscript.
    #
    # It matters beyond tidiness: multiplying by body weight inflated the
    # intake spread enormously, and since intake is the dominant term in
    # the dairy and swine equations, physiology then swamped climate and
    # the county transfer had almost nothing left to learn.
    out["dmi_kg_d"] = out["dmi_pct"]

    # A county WCC multiplies a whole inventory, so the quantity needed is
    # a HERD AVERAGE. The table's spread is between individual animals
    # across studies -- the distance from a weanling to a finished pig --
    # which is far wider than the distance between one county's average
    # and another's. Shrinking toward the mean keeps the literature's
    # central tendency while giving the spread the right meaning.
    if HERD_SPREAD != 1.0:
        for c in ("age_months", "bw_kg", "dmi_kg_d", "lm_l_d"):
            if c in out:
                m = out[c].mean()
                out[c] = m + (out[c] - m) * HERD_SPREAD
    return out


def map_to_literature_wcc(values, name: str, rep=None):
    """
    Map the equation-derived WCC onto the tabulated LEVEL.

    The published intake equations give the RESPONSE -- how water use moves
    with temperature, humidity, body weight and intake. The literature
    table gives the LEVEL. The transform below keeps the first and takes
    the second.

    IT MATCHES THE MEAN AND STANDARD DEVIATION, NOT THE FIVE QUANTILES.
    Interpolating through the tabulated quantiles reproduces them exactly,
    which sounds preferable until you look at what they are: the dairy row
    has q50 = 36.40 and q75 = 37.30, so a quarter of every sample is forced
    into a 0.9 gal/d band -- 1.6 percent of the range. Beef is tighter
    still, at 0.30 gal/d. Quantiles that close are an artifact of compiling
    a handful of commonly cited figures, not a real feature of the
    distribution, and honouring them exactly produces the dense clumps and
    empty gaps that made the sample cloud unreadable.

    Matching mean and standard deviation keeps the central tendency and the
    spread the literature reports, gives a smooth unimodal distribution,
    and leaves the ordering from the equations untouched, which is what
    carries the climate sensitivity into the regression.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if ok.sum() == 0:
        return out

    q = LITERATURE[name]["wcc_gal_d"]
    mean, std = float(q["mean"]), float(q["std"])
    if name == "poultry" and not POULTRY_WCC_USE_MEAN:
        # The tabulated poultry mean of 0.04 sits below its own 25th
        # percentile of 0.08, which no distribution can satisfy. The
        # percentiles are taken as authoritative, implying a mean near the
        # median; see the note on LITERATURE.
        mean = float(q["q50"])

    # A TRUNCATED normal, not a clipped one. The tabulated minimum sits
    # about 1.5 standard deviations below the mean for dairy, so a plain
    # normal would put seven percent of the samples past it, and clipping
    # would stack every one of them on the bound -- reintroducing exactly
    # the spike this transform exists to avoid. Truncating redistributes
    # that mass through the body of the distribution instead.
    from scipy.stats import truncnorm
    lo, hi = float(q["min"]), float(q["max"])
    a, b = (lo - mean) / std, (hi - mean) / std

    ranks = pd.Series(v[ok]).rank(method="average").to_numpy()
    probs = np.clip((ranks - 0.5) / ok.sum(), 1e-6, 1 - 1e-6)
    out[ok] = truncnorm.ppf(probs, a, b, loc=mean, scale=std) * L_PER_GAL
    return out


def _climate(rng, n):
    """
    Ambient conditions spanning the CONUS range a county-year would see.
    Minimum temperature is derived from the mean with a realistic diurnal
    range, since Murphy et al. (1983) needs Tmin while the framework's
    climate driver is mean temperature.
    """
    tmean = np.clip(rng.normal(13.0, 9.5, n), -12.0, 38.0)
    diurnal = np.clip(rng.normal(11.0, 2.5, n), 5.0, 18.0)
    tmin = tmean - diurnal / 2.0
    rh = np.clip(rng.normal(64.0, 13.0, n), 18.0, 98.0)
    wind = np.clip(rng.gamma(4.0, 2.8, n), 0.5, 35.0)          # km/h
    sun = np.clip(rng.normal(11.0, 2.0, n), 6.0, 15.0)         # hours
    return tmean, tmin, rh, wind, sun


def _noise(rng, n, cv):
    """Multiplicative lognormal residual with the requested CV, mean 1."""
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    return rng.lognormal(-0.5 * sigma ** 2, sigma, n)


def _finish(name, phys, clim, wcc_eq, rng, extra=None):
    """
    Common tail: map the equation response onto the tabulated WCC level,
    add the between-herd residual, and assemble the frame.
    """
    tmean, tmin, rh, wind, sun = clim
    wcc_exp = map_to_literature_wcc(wcc_eq, name)
    wcc = wcc_exp * _noise(rng, len(wcc_exp), RESIDUAL_CV[name])

    df = pd.DataFrame({
        "age_months": phys["age_months"], "bw_kg": phys["bw_kg"],
        "dmi_kg_d": phys["dmi_kg_d"], "dmi_pct": phys["dmi_pct"],
        "temp_c": tmean, "temp_min_c": tmin, "rh_pct": rh,
        "wind_kmh": wind, "sunlight_h": sun,
        "wcc_equation_l_d": wcc_eq,
        "wcc_expected_l_d": wcc_exp,
        "wcc_expected_gal_d": wcc_exp * GAL_PER_L,
        "wcc_l_d": wcc, "wcc_gal_d": wcc * GAL_PER_L,
    })
    if "lm_l_d" in phys:
        df["lm_l_d"] = phys["lm_l_d"]
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def gen_dairy(rng, n):
    """
    Physiology from the literature table; response from Murphy et al.
    (1983), which takes intake, milk yield, sodium and minimum temperature.
    """
    clim = _climate(rng, n)
    tmean, tmin, rh, wind, sun = clim
    phys = sample_physiology(rng, n, "dairy_cattle")

    # The table gives milk yield directly, so lactation status follows from
    # it rather than being drawn separately: an animal yielding milk is in
    # lactation by definition.
    # L in Table 2 is the lactation level, not a lactating/dry indicator.
    # The indicator is constant at 1 in this sampling space -- every
    # sampled cow is in lactation -- so it carries no variance and its
    # coefficient is unidentifiable. The tabulated milk yield is the
    # lactation level and is used directly.
    lactating = phys["lm_l_d"].to_numpy(float)
    na_g_d = np.where(lactating > 0, rng.uniform(40.0, 90.0, n),
                      rng.uniform(20.0, 50.0, n))

    eq = dairy_fwi_murphy(phys["dmi_kg_d"], phys["lm_l_d"], na_g_d, tmin)
    df = _finish("dairy_cattle", phys, clim, eq, rng,
                 {"lactating": lactating, "milk_kg_d": phys["lm_l_d"],
                  "crosscheck_l_d": dairy_fwi_nasem(
                      phys["dmi_kg_d"], rng.uniform(42, 58, n),
                      rng.uniform(220, 480, n), rng.uniform(14, 19, n), tmean)})
    return df


def gen_beef(rng, n):
    """
    Physiology from the table; response from Winchester and Morris (1956),
    WI = DMI x (3.413 + 0.01595 exp(0.17596 T)).

    Winchester rather than NASEM Eq 19-105 because the MLR is specified on
    physiological predictors. The NASEM equation takes shrunk body weight
    and an effective temperature index and does not contain intake at all,
    so a regression of it on age, body weight and intake cannot recover a
    meaningful intake coefficient -- fitted against it, the intake term
    came out at -0.005 with an R2 of 0.0002. Winchester is intake-driven
    and is the standard relation for beef water use, so the fitted
    coefficients carry their intended physiological meaning. NASEM is
    retained as the cross-check.
    """
    clim = _climate(rng, n)
    tmean, tmin, rh, wind, sun = clim
    phys = sample_physiology(rng, n, "beef_cattle")

    eq = beef_wi_winchester(phys["dmi_kg_d"], tmean)
    return _finish("beef_cattle", phys, clim, eq, rng,
                   {"ceti": ceti(tmean, rh, wind, sun),
                    "crosscheck_l_d": beef_wi_nasem(
                        phys["bw_kg"], tmean, rh, wind, sun)})


def gen_hogs(rng, n):
    """
    Physiology from the table; response from intake times the
    water-to-feed ratio, which is how the swine literature reports it.
    """
    clim = _climate(rng, n)
    tmean, tmin, rh, wind, sun = clim
    phys = sample_physiology(rng, n, "hogs")

    # Breeding share of the county inventory: reported by the census, so an
    # observable attribute rather than noise. Tied to the tabulated milk
    # yield, since a sow yielding milk is by definition breeding stock.
    breeding = np.clip((phys["lm_l_d"] / 20.0).to_numpy(), 0.0, 1.0)

    # Lactation term for swine (L in Table 2). Roughly a third of the
    # breeding herd is in lactation at any time, and a lactating sow drinks
    # far more than any other class. This is the swine analogue of the
    # dairy lactation term; the breeding share alone is not lactation.
    LACTATING_SHARE = 0.35
    lactating_sows = breeding * LACTATING_SHARE

    wfr = swine_water_feed_ratio(tmean) * (1.0 + 0.55 * lactating_sows)
    eq = phys["dmi_kg_d"].to_numpy() * wfr
    return _finish("hogs", phys, clim, eq, rng,
                   {"breeding": breeding, "lactating_sows": lactating_sows,
                    "water_feed_ratio": wfr,
                    "crosscheck_l_d": phys["dmi_kg_d"] * 2.5})


def gen_poultry(rng, n, broiler_fraction=BROILER_FRACTION):
    """
    Physiology from the table; response from the layer water-to-feed curve
    and the Pesti et al. (1985) broiler relation, mixed by flock share.
    """
    clim = _climate(rng, n)
    tmean, tmin, rh, wind, sun = clim
    phys = sample_physiology(rng, n, "poultry")

    # Layer share of the county flock: census-reported, so observable.
    # Heavier birds at a given age are layers rather than broilers, which
    # is what the tabulated weight spread of 0.2 to 3.5 kg encodes.
    layer = np.clip(rng.normal(1.0 - broiler_fraction, 0.22, n), 0.02, 0.98)

    fi = phys["dmi_kg_d"].to_numpy()
    wfr = layer * layer_water_feed_ratio(tmean) + (1.0 - layer) * 1.77
    heat = broiler_coefficient(tmean) / 5.28
    eq = fi * wfr * heat
    return _finish("poultry", phys, clim, eq, rng,
                   {"layer": layer, "crosscheck_l_d": fi * 2.0})


GENERATORS = {"dairy_cattle": gen_dairy, "beef_cattle": gen_beef,
              "hogs": gen_hogs, "poultry": gen_poultry}


# ---------------------------------------------------------------------------
# MLR
# ---------------------------------------------------------------------------

def fit_mlr(rep: Report, name: str, df: pd.DataFrame,
            test_size: float, rng_seed: int) -> Tuple[dict, dict, object]:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    feats = FEATURES[name]
    X = df[feats].to_numpy(dtype=float)
    # Fitted against the realised herd average. Now that one row IS a
    # county-year herd mean rather than an individual animal, the residual
    # is between-herd spread in management, genetics and housing -- real,
    # unexplained by the features, and the right thing for the surrogate to
    # be scored against. wcc_expected_l_d holds the noise-free equation
    # value and is what the regression figure plots on its x axis.
    y = df["wcc_l_d"].to_numpy(dtype=float)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=rng_seed)

    model = LinearRegression().fit(Xtr, ytr)
    pred_tr, pred_te = model.predict(Xtr), model.predict(Xte)

    coefs = {"intercept": float(model.intercept_)}
    coefs.update({f: float(c) for f, c in zip(feats, model.coef_)})

    stats = {
        "livestock_type": name,
        "n_total": len(df), "n_train": len(ytr), "n_test": len(yte),
        "r2_train": r2_score(ytr, pred_tr),
        "r2_test": r2_score(yte, pred_te),
        "rmse_test_l_d": float(np.sqrt(mean_squared_error(yte, pred_te))),
        "mae_test_l_d": float(np.mean(np.abs(yte - pred_te))),
        "rmspe_test_pct": float(100 * np.sqrt(np.mean(
            ((yte - pred_te) / yte) ** 2))),
        # A model predicting the training mean everywhere. Any useful model
        # must beat this, and reporting it stops a high R2 from being read
        # as skill when the target has little spread.
        "rmse_baseline_mean_l_d": float(np.sqrt(np.mean((yte - ytr.mean()) ** 2))),
        # Highest R2 any model could reach given the residual deliberately
        # injected into the samples. A fit sitting at this ceiling has
        # captured everything the source equation encodes; the remaining
        # gap is the noise, not a modelling shortfall.
        "r2_noise_floor": float(
            1 - (RESIDUAL_CV[name] ** 2 * np.mean(yte ** 2)) / np.var(yte)),
        # How the surrogate scores against a single noisy realisation, which
        # is the harder and less relevant comparison, reported for context.
        "r2_vs_noisy_sample": float(r2_score(
            df["wcc_l_d"].to_numpy(dtype=float),
            model.predict(X))),
        "slope_pred_vs_actual": float(
            LinearRegression().fit(yte.reshape(-1, 1), pred_te).coef_[0]),
    }
    stats["pct_of_attainable"] = float(
        100 * stats["r2_test"] / stats["r2_noise_floor"])

    eq = f"WCC (L/d) = {coefs['intercept']:.4f}"
    for f in feats:
        c = coefs[f]
        eq += f" {'+' if c >= 0 else '-'} {abs(c):.5f}*{f}"

    rep("\n" + "-" * 78)
    rep(f"{name}")
    rep("-" * 78)
    rep(f"  {eq}")
    rep(f"  n = {stats['n_total']:,}  (train {stats['n_train']:,}, "
        f"test {stats['n_test']:,})")
    rep(f"  R2 train {stats['r2_train']:.4f} | R2 test {stats['r2_test']:.4f}"
        f" | slope of predicted vs actual {stats['slope_pred_vs_actual']:.4f}")
    rep(f"  against a single noisy realisation instead: "
        f"R2 {stats['r2_vs_noisy_sample']:.4f} "
        f"(ceiling {stats['r2_noise_floor']:.4f} -- the gap is the injected "
        f"between-animal scatter, which averages out at county scale)")
    rep(f"  RMSE test {stats['rmse_test_l_d']:.4f} L/d | "
        f"mean-baseline RMSE {stats['rmse_baseline_mean_l_d']:.4f} L/d | "
        f"skill vs baseline {100*(1 - stats['rmse_test_l_d']/stats['rmse_baseline_mean_l_d']):.1f}%")
    rep(f"  RMSPE test {stats['rmspe_test_pct']:.1f}%")

    try:
        import statsmodels.api as sm
        ols = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
        rep(f"  OLS adj. R2 {ols.rsquared_adj:.4f}; coefficient t-statistics:")
        for f, t, p in zip(["intercept"] + feats, ols.tvalues, ols.pvalues):
            rep(f"      {f:<14} t = {t:>10.2f}   p = {p:.3g}")
    except ImportError:
        rep("  (statsmodels not installed; t-statistics skipped)")

    return coefs, stats, model


def predict_wcc(coefs: dict, name: str, df: pd.DataFrame) -> np.ndarray:
    """
    Apply a fitted surrogate equation to a feature frame.

    Uses the stored coefficients rather than the sklearn object, so the
    saved equation and the saved predictions cannot drift apart: whatever
    is written to wcc_mlr_equations is exactly what produced these numbers.
    """
    out = np.full(len(df), float(coefs["intercept"]))
    for f in FEATURES[name]:
        out = out + float(coefs[f]) * df[f].to_numpy(dtype=float)
    return out


def generate_surrogate(name: str, coefs: dict, n: int, seed: int,
                       broiler_fraction: float,
                       basis: str = "drinking") -> pd.DataFrame:
    """
    An INDEPENDENT draw of physiological and climatic states, with WCC from
    the fitted surrogate equation rather than from the literature equation.
    This is the dataset the downstream county work consumes, and drawing it
    fresh keeps it out of sample with respect to the fit.
    """
    rng = np.random.default_rng(seed + 900_000
                                + zlib.crc32(name.encode("utf-8")) % 100_000)
    df = (GENERATORS[name](rng, n, broiler_fraction) if name == "poultry"
          else GENERATORS[name](rng, n))
    df = apply_basis(df, name, basis)
    df = add_interactions(name, df)
    df = df.rename(columns={"wcc_l_d": "wcc_literature_l_d",
                            "wcc_gal_d": "wcc_literature_gal_d"})
    df["wcc_mlr_l_d"] = predict_wcc(coefs, name, df)
    df["wcc_mlr_gal_d"] = df["wcc_mlr_l_d"] * GAL_PER_L
    df["livestock_type"] = name
    return df


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def qa_samples(rep: Report, name: str, df: pd.DataFrame) -> None:
    w = df["wcc_l_d"]
    lo, hi = PLAUSIBLE_L_PER_DAY[name]
    below, above = int((w < lo).sum()), int((w > hi).sum())

    rep("\n" + "-" * 78)
    rep(f"{name}  --  generated samples")
    rep("-" * 78)
    rep(f"  WCC L/d    : min {w.min():.4f}  p25 {w.quantile(.25):.4f}  "
        f"median {w.median():.4f}  p75 {w.quantile(.75):.4f}  "
        f"max {w.max():.4f}  mean {w.mean():.4f}")
    rep(f"  WCC gal/d  : median {df['wcc_gal_d'].median():.4f}  "
        f"mean {df['wcc_gal_d'].mean():.4f}")
    rep(f"  plausible range {lo}-{hi} L/d: "
        f"{below:,} below ({100*below/len(w):.2f}%), "
        f"{above:,} above ({100*above/len(w):.2f}%)  "
        f"-- reported, not clamped")

    # A spike at any single value is the failure mode that made the previous
    # implementation unusable, so it is checked explicitly.
    top = w.round(4).value_counts().head(1)
    share = 100 * top.iloc[0] / len(w)
    rep(f"  most frequent single value: {top.index[0]:.4f} L/d "
        f"({top.iloc[0]:,} rows, {share:.3f}%)"
        + ("   <-- WARNING: mass concentrated at one value" if share > 1.0 else ""))

    if "crosscheck_l_d" in df:
        c = df["crosscheck_l_d"]
        ok = c.notna() & (c > 0)
        ratio = (w[ok] / c[ok])
        rep(f"  cross-check equation ratio (primary/alternative): "
            f"median {ratio.median():.3f}  p10 {ratio.quantile(.1):.3f}  "
            f"p90 {ratio.quantile(.9):.3f}")

    rep("  driver correlations with WCC:")
    for f in BASE_FEATURES[name]:
        rep(f"      {f:<14} r = {df[f].corr(w):>6.3f}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(data_dir: str = DATA_DIR, meta_dir: str = META_DIR,
        n_samples: int = N_SAMPLES, seed: int = SEED,
        test_size: float = 0.2, broiler_fraction: float = BROILER_FRACTION,
        verbose: bool = False, qa_report: bool = False,
        basis: str = "drinking", herd_spread: Optional[float] = None
        ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    global HERD_SPREAD
    if herd_spread is not None:
        HERD_SPREAD = float(herd_spread)
    base = Path(data_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)

    # No QA file is written unless asked for. Use --verbose to see the same
    # diagnostics on the console without leaving a file behind.
    qa_path = None
    if qa_report:
        meta = Path(meta_dir).expanduser()
        meta.mkdir(parents=True, exist_ok=True)
        qa_path = meta / "wcc_qa.txt"

    rep = Report(qa_path, echo=verbose)
    samples: Dict[str, pd.DataFrame] = {}
    all_coefs, all_stats = [], []
    try:
        rep("Water consumption coefficients -- literature-based sampling "
            "and MLR surrogate")
        rep(f"tables dir: {base}")
        rep(f"run (UTC):  {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
        rep(f"pandas {pd.__version__} | numpy {np.__version__} | "
            f"python {sys.version.split()[0]}")
        rep(f"\nn_samples = {n_samples:,} per livestock type   seed = {seed}")
        rep(f"broiler fraction of the poultry mixture = {broiler_fraction:.2f}")
        rep(f"water basis = {basis}")
        rep(f"herd spread = {HERD_SPREAD} of the tabulated between-animal "
            f"spread")
        rep("DMI is read as kg/day, not as a percentage of body weight; see "
            "sample_physiology.")
        rep("\nOne row is a county-year herd or flock average, not an "
            "individual animal, because")
        rep("that is the unit a WCC multiplies. Drivers are sampled as herd "
            "means with")
        rep("between-herd spread rather than uniformly across the lifecycle.")
        rep("\nThe MLR is a linear surrogate of the published intake equations "
            "listed in")
        rep("the module docstring. It is not an independent empirical model "
            "fitted to")
        rep("field observations, and cannot be more accurate than its source "
            "equations.")
        rep(f"\nPublished worked example reproduced at import: "
            f"CETI {_ceti_chk:.2f} (published 29.1), "
            f"NASEM beef WI {_wi_chk:.2f} L/d (published 59.1).")

        rep("\n" + "=" * 78)
        rep("SAMPLE GENERATION")
        rep("=" * 78)
        bar = tqdm(LIVESTOCK, desc="Generating WCC samples", unit="type")
        for name in bar:
            bar.set_postfix_str(name)
            # zlib.crc32, not hash(): Python randomises string hashing per
            # process, so hash() would make the samples irreproducible.
            rng = np.random.default_rng(
                seed + zlib.crc32(name.encode("utf-8")) % 100_000)
            if name == "poultry":
                df = GENERATORS[name](rng, n_samples, broiler_fraction)
            else:
                df = GENERATORS[name](rng, n_samples)
            df = add_interactions(name, df)
            samples[name] = df
            qa_samples(rep, name, df)
        bar.close()

        rep("\n" + "=" * 78)
        rep("MLR SURROGATE FITS")
        rep("=" * 78)
        bar = tqdm(LIVESTOCK, desc="Fitting MLR surrogates", unit="type")
        coef_by_type = {}
        for name in bar:
            bar.set_postfix_str(name)
            coefs, stats, _ = fit_mlr(rep, name, samples[name], test_size, seed)
            coef_by_type[name] = coefs
            row = {"livestock_type": name}
            row.update(coefs)
            all_coefs.append(row)
            all_stats.append(stats)
            # Surrogate prediction on the same rows, so the literature value
            # and the surrogate value sit side by side for comparison.
            samples[name]["wcc_mlr_l_d"] = predict_wcc(coefs, name, samples[name])
            samples[name]["wcc_mlr_gal_d"] = samples[name]["wcc_mlr_l_d"] * GAL_PER_L
        bar.close()

        rep("\n" + "=" * 78)
        rep("SURROGATE DATASETS (independent draw, WCC from the fitted equation)")
        rep("=" * 78)
        surrogate = {}
        bar = tqdm(LIVESTOCK, desc="Generating surrogate WCC", unit="type")
        for name in bar:
            bar.set_postfix_str(name)
            sur = generate_surrogate(name, coef_by_type[name], n_samples,
                                     seed, broiler_fraction, basis)
            surrogate[name] = sur
            lit, mlr = sur["wcc_literature_l_d"], sur["wcc_mlr_l_d"]
            neg = int((mlr < 0).sum())
            rep(f"\n  {name}")
            rep(f"    surrogate WCC L/d : min {mlr.min():.4f}  "
                f"median {mlr.median():.4f}  max {mlr.max():.4f}  "
                f"mean {mlr.mean():.4f}")
            rep(f"    literature  L/d   : min {lit.min():.4f}  "
                f"median {lit.median():.4f}  max {lit.max():.4f}  "
                f"mean {lit.mean():.4f}")
            rep(f"    mean ratio surrogate/literature: {(mlr/lit).mean():.4f}")
            rep(f"    out-of-sample R2 against the literature equation: "
                f"{1 - ((lit-mlr)**2).sum()/((lit-lit.mean())**2).sum():.4f}")
            if neg:
                rep(f"    NOTE: {neg:,} negative predictions "
                    f"({100*neg/len(mlr):.2f}%). A linear surrogate can go "
                    f"below zero at the edges of the domain; these are "
                    f"reported, not clipped.")
        bar.close()

        rep("\n" + "=" * 78)
        rep("OUTPUT FILES")
        rep("=" * 78)
        for name, df in samples.items():
            p = base / f"wcc_samples_{name}.feather"
            df.reset_index(drop=True).to_feather(p)
            rep(f"  {p.name}  ({len(df):,} rows x {df.shape[1]} cols) "
                f"-- literature WCC plus the surrogate fitted to it")

        for name, df in surrogate.items():
            p = base / f"wcc_surrogate_{name}.feather"
            df.reset_index(drop=True).to_feather(p)
            rep(f"  {p.name}  ({len(df):,} rows x {df.shape[1]} cols) "
                f"-- independent draw, surrogate WCC")

        wide = pd.DataFrame({f"{n}_wcc_mlr_l_d": surrogate[n]["wcc_mlr_l_d"].to_numpy()
                             for n in LIVESTOCK})
        for n in LIVESTOCK:
            wide[f"{n}_wcc_mlr_gal_d"] = surrogate[n]["wcc_mlr_gal_d"].to_numpy()
        wide = wide[[c for n in LIVESTOCK
                     for c in (f"{n}_wcc_mlr_l_d", f"{n}_wcc_mlr_gal_d")]]
        wide.to_feather(base / "wcc_mlr_surrogate_all.feather")
        rep(f"  wcc_mlr_surrogate_all.feather  ({len(wide):,} rows x "
            f"{wide.shape[1]} cols) -- one column pair per livestock type")

        coef_df = pd.DataFrame(all_coefs)
        stat_df = pd.DataFrame(all_stats)
        merged = coef_df.merge(stat_df, on="livestock_type")
        merged.to_feather(base / "wcc_mlr_coefficients.feather")
        rep(f"  wcc_mlr_coefficients.feather  ({len(merged)} rows x "
            f"{merged.shape[1]} cols)")

        eqs = []
        for row in all_coefs:
            name = row["livestock_type"]
            eq = f"WCC (L/d) = {row['intercept']:.4f}"
            for f in FEATURES[name]:
                c = row[f]
                eq += f" {'+' if c >= 0 else '-'} {abs(c):.5f}*{f}"
            eqs.append({"livestock_type": name, "equation": eq})
        eq_df = pd.DataFrame(eqs)
        eq_df.to_csv(base / "wcc_mlr_equations.csv", index=False)
        eq_df.to_feather(base / "wcc_mlr_equations.feather")
        rep("  wcc_mlr_equations.csv")
        rep("  wcc_mlr_equations.feather")

        rep("\n" + "=" * 78)
        rep("SUMMARY")
        rep("=" * 78)
        rep(stat_df.to_string(index=False))
    finally:
        rep.close()

    if not verbose:
        print("Done.")
        for st in all_stats:
            print(f"  {st['livestock_type']:<14}{st['n_total']:>8,} samples  "
                  f"R2 test {st['r2_test']:.4f}  "
                  f"RMSPE {st['rmspe_test_pct']:>5.1f}%")
        print(f"  files      wcc_samples_<type>.feather      "
              f"literature WCC + surrogate")
        print(f"             wcc_surrogate_<type>.feather    "
              f"independent draw, surrogate WCC")
        print(f"             wcc_mlr_surrogate_all.feather   "
              f"wide, one column pair per type")
        print(f"             wcc_mlr_coefficients.feather / "
              f"wcc_mlr_equations.feather")
        print(f"  basis      {basis}   herd spread {HERD_SPREAD}")
        print(f"  tables     {base}")
        if qa_path is not None:
            print(f"  QA report  {qa_path}")

    return samples, pd.DataFrame(all_coefs)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Literature-based WCC sampling and MLR surrogate fitting.")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--meta-dir", default=META_DIR)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--broiler-fraction", type=float, default=BROILER_FRACTION)
    ap.add_argument("--verbose", action="store_true",
                    help="print the full diagnostics to the console")
    ap.add_argument("--herd-spread", type=float, default=HERD_SPREAD,
                    help="fraction of the tabulated between-animal spread "
                         "to keep; 1.0 = an individual animal, 0.25 = a "
                         "county herd average")
    ap.add_argument("--basis", choices=("drinking", "total"),
                    default="drinking",
                    help="drinking = published intake equations as-is; "
                         "total = also apply the service-water factors")
    ap.add_argument("--qa-report", action="store_true",
                    help="also write wcc_qa.txt into --meta-dir "
                         "(off by default; nothing is written to disk)")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    run(args.data_dir, args.meta_dir, args.n_samples, args.seed,
        args.test_size, args.broiler_fraction, args.verbose, args.qa_report,
        args.basis, args.herd_spread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())