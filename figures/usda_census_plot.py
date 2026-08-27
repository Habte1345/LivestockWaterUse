import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from pathlib import Path

REPO  = "/scratch/hdagne1/LivestockWaterUse"
DATA  = f"{REPO}/data/usda_census"
FIGS  = Path(f"{REPO}/figures");  FIGS.mkdir(parents=True, exist_ok=True)
NON_CONUS = {"02","15","60","66","69","72","78"}
LABELS = {"dairy_cattle":"Dairy cattle", "beef_cattle":"Beef cattle",
          "poultry":"Poultry", "hogs":"Hogs"}

_counties = None
def load_counties():
    global _counties
    if _counties is None:
        g = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
                          "cb_2018_us_county_20m.zip")[["GEOID","NAME","STATEFP","geometry"]]
        _counties = g[~g["STATEFP"].isin(NON_CONUS)]
    return _counties

def plot(livestock):
    df = pd.read_feather(f"{DATA}/usda_{livestock}_county_2002_2022.feather")
    df["fips"] = df["fips"].astype(str)
    df["head"] = pd.to_numeric(df["head"], errors="coerce")
    years = sorted(df["year"].unique())

    counties = load_counties()

    # One colour scale across all panels so the years are comparable.
    pos = df.loc[df["head"] > 0, "head"]
    norm = LogNorm(vmin=max(pos.min(), 1), vmax=pos.max())

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    axes = axes.ravel()

    for ax, year in zip(axes, years):
        g = counties.merge(df.loc[df["year"] == year, ["fips","head"]],
                           left_on="GEOID", right_on="fips", how="left")
        has     = g[g["head"].notna() & (g["head"] > 0)]
        missing = g[g["head"].isna()]

        ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)

        if not missing.empty:
            missing.plot(ax=ax, transform=ccrs.PlateCarree(),
                         color="lightgrey", edgecolor="none", zorder=5)
        has.plot(ax=ax, transform=ccrs.PlateCarree(), column="head",
                 cmap="bwr", norm=norm, edgecolor="none", zorder=5)

        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.set_title(str(year), loc='center')

    # Five census years in a 2 x 3 grid, so the last panel is blank.
    for ax in axes[len(years):]:
        ax.axis("off")

    sm = ScalarMappable(cmap="bwr", norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), label="Head",
                 orientation="vertical", shrink=0.65, aspect=25, pad=0.02, extend="max")
    fig.suptitle(f'{LABELS[livestock]}: head by county across CONUS',
                 fontsize=16, fontweight="bold", y=1.02)
    # plt.savefig(FIGS / f"{livestock}_head_by_county_all_years.png",
    #             dpi=150, bbox_inches="tight")
    # plt.tight_layout()
    plt.show()

for lv in ["dairy_cattle", "beef_cattle", "poultry", "hogs"]:
    plot(lv)