"""
plot_flux_panels.py

County-level spatiotemporal patterns from the gridded HLWU dataset:
WCC, WC, the WC-to-WW consumption ratio, and WW, each as a 2 x 3 grid of
benchmark years, on a cartopy Lambert Conformal basemap.

The figure is DISPLAYED. Nothing is written to disk.

Reads an xarray Dataset with dimensions (time, lat, lon).

WC and WW are drawn only for years with a census-supported population;
the WCC and the ratio need none and are drawn for whichever benchmark
years are requested. Requesting a year without a population for WC or WW
raises rather than plotting an empty panel.

Usage
-----
    from plot_flux_panels import plot
    plot(FluxData, "Residual")
    plot(FluxData, "dairy_Wccs_adjusted", years=[1985, 1990, 1995, 2000, 2010, 2015])
    plot(FluxData, "CL_WW", cmap="jet")
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features
from affine import Affine
import pandas as pd
import pandas as pd, glob

data_dir = r'/scratch/hdagne1/LivestockWaterUse/results/netCDF'
FluxData =  xr.open_dataset(data_dir + '/County_Level_Dairy_Cattle_WC_WW_1985_2022_geo_USGS_WC_WW.nc', engine='netcdf4')
FluxData

CONUS_EXTENT = [-125, -66, 24, 50]
CENSUS_YEARS = (2002, 2007, 2012, 2017, 2022)

# Sensible defaults per variable: label, colormap, and whether the scale
# should be symmetric about zero. A residual has a sign, so its colour must
# read as direction; a magnitude does not.
VARIABLES = {
    "dairy_Wccs_adjusted": ("WCC [gal head$^{-1}$ day$^{-1}$]", "bwr", False),
    "CL_WC": ("WC [Mgal day$^{-1}$]", "bwr", False),
    "CL_WW": ("WW [Mgal day$^{-1}$]", "bwr", False),
    "CL_ratio": ("Ratio [-]", "bwr", False),
    "CL_cons_ratio_pred": ("Predicted ratio [-]", "bwr", False),
    "CL_cons_ratio_USGS": ("USGS ratio [-]", "bwr", False),
    "Residual": ("Residual [-]", "bwr", True),
    "VALUE": ("Value", "bwr", False),
}

# Variables that require a livestock population, and therefore exist only
# for census-supported years.
NEEDS_POPULATION = {"CL_WC", "CL_WW"}


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


def make_norm(values, symmetric: bool, clip_pct: float):
    """
    Continuous linear scale over the trimmed range, shared by every panel
    so the years are comparable with one another. Trimmed at the given
    percentile because a handful of extreme counties would otherwise
    compress everything else into a narrow slice of the ramp.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return mcolors.Normalize(0, 1)
    if symmetric:
        lim = float(np.nanpercentile(np.abs(v), 100 - clip_pct)) or 1.0
        return mcolors.Normalize(vmin=-lim, vmax=lim)
    lo, hi = np.nanpercentile(v, [clip_pct, 100 - clip_pct])
    if lo >= hi:
        lo, hi = v.min(), max(v.max(), v.min() + 1e-9)
    return mcolors.Normalize(vmin=lo, vmax=hi)


def plot(ds, variable: str = "Residual",
         years: Optional[Sequence[int]] = None,
         cmap: Optional[str] = None,
         clip_pct: float = 2.0, ncol: int = 3,
         label: Optional[str] = None, symmetric: Optional[bool] = None):
    """Draw the panels and return the figure. Nothing is saved."""
    if variable not in ds:
        raise KeyError(f"{variable} not in the dataset. "
                       f"Available: {list(ds.data_vars)}")

    default_label, default_cmap, default_sym = VARIABLES.get(
        variable, (variable, "jet", False))
    label = label or default_label
    cm = plt.get_cmap(cmap or default_cmap)
    symmetric = default_sym if symmetric is None else symmetric

    available = ds.time.dt.year.values
    if years is None:
        years = [y for y in (1985, 1990, 1995, 2000, 2010, 2015)
                 if y in available]

    # A population-dependent variable has no value before the first census.
    if variable in NEEDS_POPULATION:
        bad = [y for y in years if y < min(CENSUS_YEARS)]
        if bad:
            raise ValueError(
                f"{variable} requires a livestock population, which the "
                f"census provides from {min(CENSUS_YEARS)}. Requested years "
                f"without one: {bad}. Choose years within "
                f"{min(CENSUS_YEARS)}-{max(CENSUS_YEARS)}, or plot the WCC "
                f"or ratio instead, neither of which needs a population.")

    missing = [y for y in years if y not in available]
    if missing:
        raise ValueError(f"years not in the dataset: {missing}")

    idx = [int(np.where(available == y)[0][0]) for y in years]
    data = ds[variable].values
    norm = make_norm(data[idx], symmetric, clip_pct)

    lon2d, lat2d = np.meshgrid(ds.lon.values, ds.lat.values)
    nrow = int(np.ceil(len(years) / ncol))
    fig = plt.figure(figsize=(14, 4), dpi=200)
    fig.subplots_adjust(wspace=0.02, hspace=0.25)

    for i, (t, year) in enumerate(zip(idx, years), start=1):
        ax = fig.add_subplot(nrow, ncol, i,
                             projection=ccrs.LambertConformal())
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        ax.set_aspect("auto")

        ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
        ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                       edgecolor="#8fa8b8", linewidth=0.4)

        ax.pcolormesh(lon2d, lat2d, data[t], cmap=cm, norm=norm,
                      shading="auto", transform=ccrs.PlateCarree(),
                      zorder=5)

        ax.add_feature(cfeature.STATES, edgecolor="#888888",
                       linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
                       linewidth=0.7, zorder=6)
        ax.set_title(f"{year}", fontsize=13, fontweight="bold")

        # Distribution inset, lower right, rotated so its value axis runs
        # parallel with the colourbar and the two scales read together.
        v = data[t].ravel()
        v = v[np.isfinite(v)]
        if v.size:
            ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
            lim_lo, lim_hi = norm.vmin, norm.vmax
            ins.hist(np.clip(v, lim_lo, lim_hi), bins=30, color="#c1c97d",
                     edgecolor="black", linewidth=0.4,
                     orientation="horizontal")
            mean_val = float(np.mean(v))
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

        print(f"  {year}  n={v.size:,}  mean {np.mean(v):+.4g}  "
              f"median {np.median(v):+.4g}  "
              f"range [{np.min(v):.4g}, {np.max(v):.4g}]")

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.95, aspect=30, pad=0.02,
                      extend="both")
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=9)

    _show(fig)
    plt.close(fig)
    return fig