# High-Resolution County-Level Livestock Water Use for the CONUS

Estimates of livestock water **consumption (WC)** and **withdrawal (WW)** for
every county in the conterminous United States, for four livestock
categories, built from published animal-physiology intake equations, USDA
agricultural census inventories, and ERA5-Land climate.

The framework replaces a single national coefficient per animal with a
county- and climate-specific one, so that a dairy cow in Texas and one in
Minnesota are not assumed to drink the same amount.

Livestock categories: **dairy cattle, beef cattle, hogs, poultry**
(layers + broilers combined).

---

## 1. What the model does

Water consumption for a county is

```
WC = WCC × head count
```

where `WCC` is the water consumption coefficient in L head⁻¹ day⁻¹ and the
head count comes from the USDA Census of Agriculture. Withdrawal follows
from the consumptive fraction:

```
WW = WC / (WC:WW ratio)
```

Two components have to be estimated at county level. Each is a pathway.

### Pathway 1 — county WCC from climate

1. **Literature-based sampling.** Physiological states (body weight, dry
   matter intake, age, lactation or breeding status) and climate are
   sampled together over literature-constrained ranges, and the WCC for
   each sample is computed from a **published intake equation**, not an
   invented multiplier stack:

   | Type | Equation | Source |
   |---|---|---|
   | Dairy | FWI = 15.99 + 1.58·DMI + 0.90·MY + 0.05·Na + 1.20·T<sub>min</sub> | Murphy et al. (1983), adopted by NRC (2001) |
   | Beef | WI = f(SBW, CETI), quadratic in both | NASEM (2016) Eq 19-105, R² = 0.997 |
   | Hogs | WI = water:feed ratio × feed intake | Shaw et al. (2006); K-State Swine Nutrition Guide |
   | Broiler | WI (mL/d) = 5.28 × age in days | Pesti et al. (1985), R² > 0.99 |
   | Layer | WI = water:feed ratio × feed intake | Lohmann layer management guidance |

   Cross-checks: Winchester & Morris (1956) for beef, NASEM (2021) Eq 9-1
   for dairy. The NASEM beef equation and its CETI index are asserted at
   import against the published worked example (CETI 29.10, WI 59.09 L/d);
   the module refuses to run if they do not reproduce.

2. **MLR surrogate.** A multiple linear regression is fitted per livestock
   type over those samples, giving four generic WCC equations. Interaction
   and quadratic terms are taken from the *structure* of the source
   equations (CETI is quadratic in temperature; broiler intake is
   c(T)·age), not selected by searching for fit.

3. **ANN transfer to counties.** The MLR is evaluated over sampled
   physiology and climate to produce generic WCC values; an ANN is then
   trained on the **climate columns only**, with observed county herd
   composition (layer share, breeding share) as additional inputs where
   the census reports it. Applying that network to county climate gives a
   county-specific WCC.

### Pathway 2 — county WC:WW ratio from climate

USGS publishes a livestock consumptive-use ratio at state level
(1960–1980) and county level (1985–1995). Two designs were tested for
transferring it to all counties and years — see **Results** below for what
each achieved.

---

## 2. Repository layout

```
scripts/                     the pipeline, run in this order
  extract_usgs_livestock.py    USGS water-use, state 1960-1980 + county 1985-2015
  usda_livestock.py            USDA census head counts + herd composition
  era5_climate_GEE.py          ERA5-Land monthly forcings via Earth Engine
  wcc_mlr.py                   literature WCC samples -> four MLR equations
  wcc_ann_downscale.py         generic WCC -> county WCC  (pathway 1)
  ratio_ann_downscale.py       state ratio -> county ratio (pathway 2)
  county_level_WC.py           WC = WCC x head count
  validate_against_usgs.py     WW = WC / ratio, compared with USGS

figures/                     display-only plotting, nothing written to disk
  usgs_livestock_wateruse_plot.py   USGS source data
  usda_census_plot.py               census inventories
  climate_factors_plot.py           ERA5 county maps
  mlr_plot.py                       MLR surrogate vs literature WCC
  wcc_ANN_plot.py                   ANN transfer performance
  wcc_county_based_plot.py          county WCC maps
  plot_wc_maps.py                   county WC maps
  plot_ratio_ann.py                 ratio downscaling performance
  plot_validation_residuals.py      residuals against USGS, by year

data/                        all outputs, Feather format, gitignored
notebook/                    exploratory work
results/                     final figures and tables for the manuscript
```

`data/` is not tracked in git — it is regenerable by running the pipeline
and is far too large for a repository.

---

## 3. Running the pipeline

Order matters: each step reads the previous step's tables.

```bash
python scripts/extract_usgs_livestock.py
python scripts/usda_livestock.py
python scripts/era5_climate_GEE.py --project <earth-engine-project>

python scripts/wcc_mlr.py                     # must precede the ANN step
python scripts/wcc_ann_downscale.py
python scripts/ratio_ann_downscale.py

python scripts/county_level_WC.py --interpolate
python scripts/validate_against_usgs.py
```

`--interpolate` fills the years between agricultural censuses. Without it
the modelled series exists only for 2002, 2007, 2012, 2017 and 2022, which
do not overlap the USGS withdrawal years, and the validation step has
nothing to compare.

Every script takes `--help`. All accept `--data-dir` and related overrides
so a run can be directed elsewhere without editing the source.

---

## 4. Data sources

| Source | What | Coverage |
|---|---|---|
| USGS Water Use | livestock withdrawal, consumptive use, WC:WW ratio | state 1960–1980; county 1985–2015 (consumption ends 1995) |
| USDA Census of Agriculture | county head counts by class | 2002, 2007, 2012, 2017, 2022 |
| ERA5-Land Monthly Aggregated (GEE) | temperature, dewpoint, wind, radiation, precipitation | 1960–2022, monthly |
| Census TIGER / cartographic boundaries | county and state geometry | 2018 vintage |

All tables key on a **5-character county FIPS**, so they join directly to
one another.

Derived climate variables: relative humidity from Magnus-Tetens computed
**per month** then averaged (RH is nonlinear in temperature, so the mean of
monthly RH is not RH of the annual means); 2 m wind via the FAO-56 log
profile; sunshine hours by inverting the Ångström–Prescott relation
against ERA5 radiation; and CETI from all four, per NASEM (2016).

---

## 5. Results

### Pathway 1 — WCC transfer (ANN, held-out test)

| Type | R² | Climate-only ceiling | % of ceiling reached |
|---|---|---|---|
| Dairy cattle | 0.73 | 0.73 | ~99% |
| Beef cattle | 0.97 | 0.89 | — |
| Hogs | 0.79 | 0.67 | — |
| Poultry | 0.84 | 0.52 | — |

Beef is the strongest because CETI is genuinely climate-driven. Dairy sits
at its attainable ceiling: the remaining error is between-herd variation
that no climate input can resolve.

### Pathway 2 — WC:WW ratio downscaling

Two designs were tested, both against the baseline of applying the state
ratio to its counties unchanged.

| Design | Validation | Result |
|---|---|---|
| State-level ANN (1960–1980 training) | grouped CV by state | ANN RMSE 0.151 vs linear 0.146 vs constant-1.0 0.174 |
| Anomaly model (county ratio as target) | held out by state | best model RMSE 0.0868 vs **state base 0.0864** |

**Neither design improves on the state ratio applied directly.** The
reason is in the data: 67% of county ratios are exactly 1.0, because USGS
assumed near-complete consumption for most areas. The quantity is largely
a reporting convention rather than a climate-driven physical variable, so
there is little spatial signal to downscale.

### End-to-end validation against USGS

Withdrawal is the decisive test — it exercises the whole chain at once
against observations the model never saw.

| Comparison | n | R² | RMSE | PBIAS |
|---|---|---|---|---|
| **Withdrawal, log₁₀** (2005/2010/2015) | 8,931 | **0.786** | 0.261 | — |
| Withdrawal, raw | 9,151 | 0.581 | 0.908 Mgal/d | **−28.6%** |
| Ratio (1985/1990/1995) | 8,893 | 0.560 | 0.139 | −0.2% |

The log-space R² of 0.786 across 3,051 counties says the **spatial pattern
is reproduced well**. The −28.6% bias is one-directional and physically
explicable: the published equations give *drinking* water, whereas USGS
livestock withdrawal also includes service water — parlour and pen
washing, evaporative cooling, drinker spillage. A shortfall of roughly a
quarter is the size that gap would produce.

---

## 6. Known limitations

These belong in any write-up of this dataset.

- **County consumptive use cannot be validated directly.** USGS stopped
  collecting it after 1995; the modelled series begins in 2002 with the
  first census head counts. WC is tested only indirectly, through WW.
- **The ratio validation is circular.** For 1985–1995 the model's
  `state_ratio` input is derived from the mean of the same county ratios
  it is scored against, because the published state series ends in 1980.
  The R² of 0.560 must not be reported as independent validation.
- **Post-1995 ratios rest on an assumption.** Each state's last observed
  ratio is carried forward, labelled `carried from the nearest observed
  year` in `base_source`.
- **The MLR is a surrogate, not an empirical model.** It is a linear
  approximation of published intake equations and cannot be more accurate
  than they are. It should be described as such.
- **Head counts between census years are interpolated**, and the
  validation years 2005/2010/2015 all fall between censuses. `head_origin`
  records `census` or `interpolated` per row. The withdrawal comparison
  therefore tests the model *and* the interpolation together.
- **TIGER 2018 geometry is applied to all years back to 1960.** County
  boundaries changed over that period.
- **Systematic −28.6% low bias** against USGS withdrawal, attributed to
  service water. A `SERVICE_WATER_FACTOR` exists in `wcc_mlr.py` but is
  **disabled by default** and marked as an assumption pending cited
  values.
- **444 hog counties and some poultry counties have no WCC**, where county
  climate is incomplete. These propagate as NaN rather than zero, and
  counts are reported at each step.

---

## 7. Requirements

```
python >= 3.10
pandas  numpy  pyarrow  scipy  scikit-learn  statsmodels
matplotlib  seaborn  cartopy  geopandas
earthengine-api          # ERA5-Land extraction only
requests  openpyxl  xlrd  tqdm
```

Earth Engine requires an authenticated Cloud project; pass it with
`--project`.

---

## 8. Conventions used throughout

- **Feather** for every table, keyed on 5-character FIPS.
- **Nothing is zero-filled.** Suppressed, missing or unresolvable values
  stay NaN and are counted in the run summary. A county with no climate
  gets no WCC, not a fabricated one.
- **Monthly before annual.** The water-to-feed response is convex in
  temperature, so evaluating a coefficient on an annual mean understates
  water use by 4–9% depending on the seasonal range. Coefficients are
  evaluated monthly and aggregated afterwards.
- **Rates average, volumes sum.** WCC and WC are daily rates, so months
  average; annual volumes multiply by days-in-month first, then sum.
- **Plots display, they do not save.** Figure scripts render inline and
  write nothing unless explicitly asked.
