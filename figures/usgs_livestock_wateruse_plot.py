import geopandas as gpd, pandas as pd, numpy as np, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

REPO = "/scratch/hdagne1/LivestockWaterUse"
USGS = f"{REPO}/data/usgs_ls_state_county_1960_2015/usgs_livestock_county_1985_2015.feather"
FIGS = Path(f"{REPO}/figures"); FIGS.mkdir(parents=True, exist_ok=True)
NON_CONUS = {"02","15","60","66","69","72","78"}
CMAP = "bwr"

VARS = {
    "ls_withdrawal_fresh_mgd":  ("Livestock withdrawal", "Mgal/d", "log"),
    "ls_consumptive_fresh_mgd": ("Livestock consumptive use", "Mgal/d", "log"),
}

_counties = None
def load_counties():
    global _counties
    if _counties is None:
        g = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
                          "cb_2018_us_county_20m.zip")[["GEOID","NAME","STATEFP","geometry"]]
        _counties = g[~g["STATEFP"].isin(NON_CONUS)]
    return _counties

def plot(var):
    label, unit, kind = VARS[var]
    df = pd.read_feather(USGS)
    df["fips"] = df["fips"].astype(str)
    df[var] = pd.to_numeric(df[var], errors="coerce")

    years = sorted(y for y in df["year"].unique()
                   if df.loc[df["year"] == y, var].notna().any())
    counties = load_counties()

    if kind == "ratio":
        plot_col, vals = var, df[var].dropna()
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=1.001)
        ticks = [0, 0.25, 0.5, 0.75, 1.0, norm.vmax]
        fmt = lambda t: f"{t:.2f}"
    else:
        # Values span orders of magnitude, so scale in log space first and
        # centre the colormap on the median county. TwoSlopeNorm cannot be
        # combined with LogNorm, hence the explicit log10 column.
        plot_col = f"_{var}_log10"
        pos = df.loc[df[var] > 0, var]
        df[plot_col] = np.log10(df[var].where(df[var] > 0))
        lo, mid, hi = (np.log10(pos.quantile(0.01)),
                       np.log10(pos.median()),
                       np.log10(pos.quantile(0.99)))
        norm = TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)
        ticks = np.linspace(lo, hi, 6)
        fmt = lambda t: f"{10**t:,.2f}"

    n = len(years)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    for ax, year in zip(axes, years):
        g = counties.merge(df.loc[df["year"] == year, ["fips", plot_col]],
                           left_on="GEOID", right_on="fips", how="left")
        has     = g[g[plot_col].notna()]
        missing = g[g[plot_col].isna()]

        ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)

        if not missing.empty:
            missing.plot(ax=ax, transform=ccrs.PlateCarree(),
                         color="lightgrey", edgecolor="none", zorder=5)
        has.plot(ax=ax, transform=ccrs.PlateCarree(), column=plot_col,
                 cmap=CMAP, norm=norm, edgecolor="none", zorder=5)

        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.set_title(str(year), loc='center')

    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="vertical",
                        shrink=0.65, aspect=25, pad=0.02, extend="both",
                        ticks=ticks)
    cbar.ax.set_yticklabels([fmt(t) for t in ticks])
    cbar.set_label(f"{label} ({unit})")
    fig.suptitle(f'USGS {label} by county across CONUS',
                 fontsize=16, fontweight="bold", y=1.02)
    plt.show()

for v in VARS:
    plot(v)