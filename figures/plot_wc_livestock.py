"""
plot_wc_livestock.py

County water consumption for one year, one panel per livestock type, on a
cartopy Lambert Conformal basemap.

The figure is DISPLAYED. Nothing is written to disk.

Reads county_level_WC_all_livestock.feather. Rows are county-month, so
they are averaged to one value per county: WC is a rate in Mgal/d, and the
annual figure is the mean daily rate, not the sum of twelve monthly rates.

WC exists only for census-supported years. Requesting a year without a
population raises rather than plotting empty panels.

Usage
-----
    from plot_wc_livestock import plot
    plot(2002)
    plot(2017, units="m3")
"""

from __future__ import annotations

import io
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_REPO = "/scratch/hdagne1/LivestockWaterUse"
WC_FILE = f"{_REPO}/data/wc_county/county_level_WC_all_livestock.feather"

CONUS_EXTENT = [-125, -66, 24, 50]
CENSUS_YEARS = (2002, 2007, 2012, 2017, 2022)

NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
              "cb_2018_us_county_20m.zip")

PANELS = [("dairy_cattle", "Dairy cattle"), ("beef_cattle", "Beef cattle"),
          ("hogs", "Hogs"), ("poultry", "Poultry")]

UNITS = {"mgal": ("wc_mgal_d", "WC [Mgal day$^{-1}$]"),
         "m3": ("wc_m3_d", "WC [m$^3$ day$^{-1}$]"),
         "l": ("wc_l_d", "WC [L day$^{-1}$]")}


def _show(fig, dpi: int = 130) -> None:
    """
    plt.show() is a silent no-op on a non-interactive backend, and
    display(fig) falls back to a text repr unless the inline backend
    registered its PNG formatter, so the bytes are encoded here.
    """
    try:
        import IPython
        ip = IPython.get_ipython()
    except Exception:
        ip = None
    if ip is None:
        plt.show()
        return
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    display(Image(data=buf.getvalue()))
    plt.close(fig)


def make_norm(values, clip_pct: float):
    """
    Continuous linear scale over the trimmed range. Trimmed at the given
    percentile because a handful of extreme counties would otherwise
    compress everything else into a narrow slice of the ramp.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return mcolors.Normalize(0, 1)
    lo, hi = np.nanpercentile(v, [clip_pct, 100 - clip_pct])
    if lo >= hi:
        lo, hi = v.min(), max(v.max(), v.min() + 1e-9)
    return mcolors.Normalize(vmin=lo, vmax=hi)


def load_counties(shapefile: Optional[str] = None):
    import geopandas as gpd
    g = gpd.read_file(shapefile if shapefile else COUNTY_URL)
    g = g[~g["STATEFP"].isin(NON_CONUS_FIPS)].copy()
    g["fips"] = g["GEOID"].astype(str).str.zfill(5)
    return g.to_crs("EPSG:4326")


def plot(year: int = 2002, wc_file: str = WC_FILE,
         shapefile: Optional[str] = None, cmap: str = "bwr",
         units: str = "mgal", clip_pct: float = 2.0, ncol: int = 2):
    """Draw the panels and return the figure. Nothing is saved."""
    if year < min(CENSUS_YEARS):
        raise ValueError(
            f"WC requires a livestock population, which the census provides "
            f"from {min(CENSUS_YEARS)}. Choose a year within "
            f"{min(CENSUS_YEARS)}-{max(CENSUS_YEARS)}.")

    col, label = UNITS[units]
    df = pd.read_feather(wc_file)
    df = df[df["year"] == year]
    if df.empty:
        raise ValueError(f"no rows for {year}")
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    # WC is a rate, so months average rather than sum.
    annual = (df.groupby(["livestock_type", "fips"], as_index=False)[col]
              .mean())

    counties = load_counties(shapefile)
    cm = plt.get_cmap(cmap)
    nrow = int(np.ceil(len(PANELS) / ncol))
    fig = plt.figure(figsize=(14, 4), dpi=200)
    fig.subplots_adjust(wspace=0.02, hspace=0.25)

    for i, (stem, title) in enumerate(PANELS, start=1):
        ax = fig.add_subplot(nrow, ncol, i,
                             projection=ccrs.LambertConformal())
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        ax.set_aspect("auto")

        ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
        ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                       edgecolor="#f0f2f4", linewidth=0.4)

        sub = annual[annual["livestock_type"] == stem][["fips", col]]
        g = counties.merge(sub, on="fips", how="left")
        v = g[col].to_numpy(dtype=float)

        # One scale per panel: a dairy county consumes orders of magnitude
        # more than a poultry county, so a shared scale would render three
        # of the four panels a single flat colour.
        norm = make_norm(v, clip_pct)

        g.plot(column=col, cmap=cm, norm=norm, ax=ax, edgecolor="none",
               transform=ccrs.PlateCarree(), zorder=5,
               missing_kwds={"color": "white", "edgecolor": "none"})

        ax.add_feature(cfeature.STATES, edgecolor="#888888",
                       linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
                       linewidth=0.7, zorder=6)
        ax.set_title(f"{title}", fontsize=13, fontweight="bold")

        # Distribution inset, lower right, rotated so its value axis runs
        # parallel with the colourbar and the two scales read together.
        vv = v[np.isfinite(v)]
        if vv.size:
            ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
            lim_lo, lim_hi = norm.vmin, norm.vmax
            ins.hist(np.clip(vv, lim_lo, lim_hi), bins=30, color="#c1c97d",
                     edgecolor="black", linewidth=0.4,
                     orientation="horizontal")
            mean_val = float(np.mean(vv))
            ins.axhline(np.clip(mean_val, lim_lo, lim_hi), color="red",
                        linestyle="--", linewidth=1.4)
            ins.text(1.0, 1.03, f"{mean_val:.4g}", color="blue",
                     ha="right", va="bottom", fontsize=8,
                     transform=ins.transAxes)
            ins.set_ylim(lim_lo, lim_hi)
            ins.invert_xaxis()          # bars grow away from the colourbar
            ins.set_xticks([])
            ins.set_yticks([])
            ins.patch.set_alpha(0.85)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.9, aspect=30, pad=0.02,
                          extend="both")
        cb.set_label(label, fontsize=12)
        cb.ax.tick_params(labelsize=9)

        print(f"  {title:<14}n={vv.size:,}  mean {np.mean(vv):+.4g}  "
              f"median {np.median(vv):+.4g}  "
              f"range [{np.min(vv):.4g}, {np.max(vv):.4g}]")

    fig.suptitle(f"County livestock water consumption, {year}",
                 fontsize=15, fontweight="bold")
    _show(fig)
    plt.close(fig)
    return fig