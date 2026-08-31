# LivestockWaterUse

County-level livestock water consumption and withdrawal for the
conterminous United States — dairy cattle, beef cattle, hogs and poultry.

## Install

```bash
conda activate <env>
pip install pandas numpy pyarrow scipy scikit-learn statsmodels pyyaml \
            matplotlib seaborn cartopy geopandas earthengine-api \
            requests openpyxl xlrd tqdm
```

Set your Earth Engine project in `src/config.yaml`:

```yaml
climate:
  project: your-ee-project-id
```

## Usage

One command, whatever state the repository is in:

```bash
python src/main.py                          # all four livestock types
python src/main.py --livestock dairy_cattle # one type
python src/main.py --livestock hogs poultry # a subset
```

On an empty `data/` this downloads the sources and runs everything. On a
populated one it runs only what is missing or out of date. Nothing has to
be passed to say which.

Other options:

```bash
python src/main.py --dry-run     # show the plan, run nothing
python src/main.py --list        # the stages and what each produces
python src/main.py --from wcc    # this stage onward
python src/main.py --stage ratio # one stage
python src/main.py --rebuild     # ignore what is already built
```

Settings live in `src/config.yaml`. A key in a stage block becomes a flag
on the underlying script: `n_samples: 50000` → `--n-samples 50000`,
`interpolate: true` → `--interpolate`.

## Stages

```
foundation   usgs → usda → climate → mlr      shared, built once
pathway 1    wcc → wc                         county water consumption
pathway 2    ratio                            consumption-to-withdrawal ratio
evaluate     validate                         WW = WC / ratio, vs USGS
```

Foundation stages are shared by every user and always cover all four
types; `--livestock` applies to `wcc`, `wc`, `ratio` and `validate`.

`validate` requires all four types, because USGS reports livestock as a
single category and a subset cannot be compared against it.

## What runs and what is skipped

Each stage writes a `.pipeline.json` stamp beside its output recording a
fingerprint of the script and its settings, plus the livestock types
covered. A stage is skipped only if the output exists, the fingerprint
matches, and the requested types are already covered.

Editing a script or changing its settings rebuilds that stage **and
everything downstream of it**, so a stale intermediate is never reused.

## Layout

```
src/         main.py, config.yaml
scripts/     pipeline stages
figures/     plotting, display only
data/        generated tables (gitignored)
notebook/    exploratory work
results/     final figures and tables
```

## Data

Feather throughout, keyed on 5-character county FIPS. Missing values stay
NaN rather than being zero-filled. Columns ending `_origin` or `_source`
record provenance where a value is not a direct observation.

```
data/usgs_ls_state_county_1960_2015/   USGS livestock water use
data/usda_census/                      county head counts
data/climate/                          ERA5-Land forcings
data/wcc_mlr/                          WCC equations and samples
data/wcc_county/                       county WCC
data/ratio_county/                     county WC:WW ratio
data/wc_county/                        county water consumption
data/validation/                       modelled vs observed
```

Sources: USGS National Water Use, USDA NASS Census of Agriculture,
ERA5-Land via Google Earth Engine, Census TIGER boundaries. The first two
are cached locally and skipped on later runs.

## Figures

Display inline, write nothing unless given `--save <path>`:

```python
from plot_wc_maps import plot
plot()
```

## Notes

Runs are seeded and reproducible via `seed` in the config.

A stage that fails stops the run and prints how to resume:
`--from <stage>`.

In Jupyter, import and call the function rather than using `%run`, and
restart the kernel after editing a script.

## License

See `LICENSE`.
