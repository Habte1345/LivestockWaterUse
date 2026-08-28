# LivestockWaterUse

County-level livestock water consumption and withdrawal for the
conterminous United States — dairy cattle, beef cattle, hogs and poultry.

## Install

```bash
conda activate <env>
pip install pandas numpy pyarrow scipy scikit-learn statsmodels \
            matplotlib seaborn cartopy geopandas earthengine-api \
            requests openpyxl xlrd tqdm
```

## Usage

Run in order; each step reads the previous step's tables.

```bash
python scripts/extract_usgs_livestock.py                        # USGS water use
python scripts/usda_livestock.py                                # census head counts
python scripts/era5_climate_GEE.py --project <ee-project>       # ERA5-Land climate
python scripts/wcc_mlr.py                                       # WCC equations
python scripts/wcc_ann_downscale.py                             # county WCC
python scripts/ratio_ann_downscale.py                           # county WC:WW ratio
python scripts/county_level_WC.py --interpolate                 # county WC
python scripts/validate_against_usgs.py                         # WW and validation
```

`--interpolate` is required for the final step. Every script takes
`--help`, `--data-dir` and `--verbose`.

Figures display inline and write nothing unless given `--save <path>`:

```python
from plot_wc_maps import plot
plot()
```

## Layout

```
scripts/     pipeline, run in the order above
figures/     plotting, display only
data/        generated tables (gitignored)
notebook/    exploratory work
results/     final figures and tables
```

## Data

All tables are Feather, keyed on 5-character county FIPS. Missing values
stay NaN rather than being zero-filled. Columns ending in `_origin` or
`_source` record provenance where a value is not a direct observation.

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
ERA5-Land via Google Earth Engine, Census TIGER boundaries.

## Notes

Runs are seeded and reproducible with `--seed`. Source columns and bands
are validated on load, so upstream renames fail immediately.

In Jupyter, import and call the function rather than using `%run`, and
restart the kernel after editing a script.

## License

See `LICENSE`.
