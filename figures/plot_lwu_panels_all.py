"""
plot_lwu_panels.py

County-level spatiotemporal patterns from the gridded HLWU dataset, laid
out as four stacked blocks, each a 2 x 3 grid of benchmark years with its
own colourbar:

    a) WCC    b) WC    c) consumption-ratio    d) WW

The figure is DISPLAYED. Nothing is written to disk.

Reads an xarray Dataset with dimensions (time, lat, lon).

Usage
-----
    from plot_lwu_panels import plot
    plot(FluxData)
    plot(FluxData, panel_w=2.4, panel_h=1.3)
    plot(FluxData, years=[2002, 2007, 2012, 2017, 2022])
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

CONUS_EXTENT = [-125, -66, 24, 50]

# (variable, block letter, colourbar label). Order sets the block order.
# "WCC" is a placeholder: the coefficient is named per livestock type in
# the source files (dairy_Wccs_adjusted, beef_Wccs_adjusted, ...), so it is
# resolved from whichever *_Wccs_adjusted variable the file carries.
BLOCKS = [
    ("WCC", "a", "WCC [gal/head/day]"),
    ("CL_WC", "b", "WC [Mgal/day]"),
    ("CL_ratio", "c", "Ratio [-]"),
    ("CL_WW", "d", "WW [Mgal/day]"),
]

DEFAULT_YEARS = (1985, 1990, 1995, 2000, 2010, 2015)

NETCDF_DIR = "/scratch/hdagne1/LivestockWaterUse/results/netCDF"

# File stem and display name per livestock type. The stems are not uniform
# in the archive -- hogs carries an extra "County" -- so they are listed
# rather than constructed.
LIVESTOCK = [
    ("Dairy_Cattle", "Dairy cattle"),
    ("Beef_Cattle", "Beef cattle"),
    ("Hogs_County", "Hogs"),
    ("Poultry", "Poultry"),
]


def wcc_name(ds) -> str:
    """The per-type WCC variable, whatever it is called in this file."""
    for v in ds.data_vars:
        if str(v).endswith("_Wccs_adjusted"):
            return str(v)
    raise KeyError(f"no *_Wccs_adjusted variable in {list(ds.data_vars)}")


def plot_all(netcdf_dir: str = NETCDF_DIR, **kw):
    """
    One figure per livestock type, drawn in sequence.

    Each file names its coefficient differently, so the WCC block is
    resolved per file rather than assumed.
    """
    import xarray as xr
    figs = {}
    for stem, title in LIVESTOCK:
        path = (f"{netcdf_dir}/County_Level_{stem}"
                f"_WC_WW_1985_2022_geo_USGS_WC_WW.nc")
        ds = xr.open_dataset(path)
        print(f"\n{title}")
        figs[title] = plot(ds, title=title, **kw)
    return figs


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


def plot(ds, blocks: Sequence[Tuple[str, str, str]] = BLOCKS,
         years: Optional[Sequence[int]] = None, cmap: str = "bwr",
         panel_w: float = 2.6, panel_h: float = 1.35, ncol: int = 3,
         wspace: float = 0.02, hspace: float = 0.25,
         block_gap: float = 0.045, title: Optional[str] = None):
    """
    Draw the stacked blocks and return the figure. Nothing is saved.

    panel_w and panel_h set the size of ONE map panel in inches; the figure
    size follows from them and the number of blocks. Changing them changes
    only the figure's width and height, because the maps use
    set_aspect("auto") and stretch to fill whatever box they are given, so
    no gaps open between columns when the height is reduced.
    """
    years = list(years or [y for y in DEFAULT_YEARS
                           if y in ds.time.dt.year.values])
    available = ds.time.dt.year.values
    missing = [y for y in years if y not in available]
    if missing:
        raise ValueError(f"years not in the dataset: {missing}")
    idx = [int(np.where(available == y)[0][0]) for y in years]

    # Resolve the per-type WCC name before selecting the blocks.
    blocks = [(wcc_name(ds), l, c) if v == "WCC" else (v, l, c)
              for v, l, c in blocks]
    present = [b for b in blocks if b[0] in ds]
    if not present:
        raise KeyError(f"none of {[b[0] for b in blocks]} in the dataset. "
                       f"Available: {list(ds.data_vars)}")

    nrow = int(np.ceil(len(years) / ncol))
    cm = plt.get_cmap(cmap)

    lat = ds.lat.values
    lon = ds.lon.values
    # imshow, not pcolormesh. The grid is regular, and pcolormesh would
    # draw millions of quadrilaterals per panel; imshow sends one image.
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    origin = "upper" if lat[0] > lat[-1] else "lower"

    fig = plt.figure(figsize=(panel_w * ncol + 0.9,
                              panel_h * nrow * len(present)
                              + 0.35 * len(present)), dpi=200)
    # Headroom for the title, so it cannot overlap the first row of maps.
    top_margin = 0.965 if title else 1.0

    # Each block gets its own sub-grid and its own colourbar, since the
    # four quantities differ by orders of magnitude and share no scale.
    block_h = ((top_margin - block_gap * (len(present) - 1))
               / len(present))

    for bi, (var, letter, cbl) in enumerate(present):
        top = top_margin - bi * (block_h + block_gap)
        gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.88,
                              top=top, bottom=top - block_h,
                              wspace=wspace, hspace=hspace)

        data = ds[var].values
        finite = data[idx][np.isfinite(data[idx])]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0

        axes = []
        for k, (t, year) in enumerate(zip(idx, years)):
            ax = fig.add_subplot(gs[k // ncol, k % ncol],
                                 projection=ccrs.LambertConformal())
            ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
            ax.set_aspect("auto")

            ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
            ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
            ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                           edgecolor="#f0f2f4", linewidth=0.3)

            im = ax.imshow(data[t], cmap=cm, vmin=vmin, vmax=vmax,
                           extent=extent, origin=origin,
                           transform=ccrs.PlateCarree(), zorder=5,
                           interpolation="nearest")

            ax.add_feature(cfeature.STATES, edgecolor="#888888",
                           linewidth=0.35, zorder=6)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
                           linewidth=0.5, zorder=6)
            ax.set_title(f"{year}", fontsize=9)
            axes.append(ax)

        cb = fig.colorbar(im, ax=axes, shrink=0.95, aspect=26, pad=0.015,
                          fraction=0.03)
        cb.set_label(cbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Block letter, outside the first panel on the left.
        axes[0].text(-0.10, 1.18, f"{letter})", transform=axes[0].transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

        v = data[idx][np.isfinite(data[idx])]
        print(f"  {letter}) {var:<22} range [{v.min():.4g}, {v.max():.4g}]  "
              f"mean {v.mean():.4g}")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    _show(fig)
    plt.close(fig)
    return fig