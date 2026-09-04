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
    plot(FluxData, "CL_WW", cmap="bwr")

    from plot_flux_panels import plot_wcc, plot_ratio, plot_residual
    plot_wcc()          # one figure per livestock type
    plot_ratio()
    plot_residual()
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

CONUS_EXTENT = [-125, -66, 24, 50]
CENSUS_YEARS = (2002, 2007, 2012, 2017, 2022)

# Sensible defaults per variable: label, colormap, and whether the scale
# should be symmetric about zero. A residual has a sign, so its colour must
# read as direction; a magnitude does not.
VARIABLES = {
    "dairy_Wccs_adjusted": ("WCC [gal head$^{-1}$ day$^{-1}$]", "bwr", False),
    "beef_Wccs_adjusted": ("WCC [gal head$^{-1}$ day$^{-1}$]", "bwr", False),
    "hogs_Wccs_adjusted": ("WCC [gal head$^{-1}$ day$^{-1}$]", "bwr", False),
    "poultry_Wccs_adjusted": ("WCC [gal head$^{-1}$ day$^{-1}$]", "bwr", False),
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
         label: Optional[str] = None, symmetric: Optional[bool] = None,
         title: Optional[str] = None):
    """Draw the panels and return the figure. Nothing is saved."""
    if variable not in ds:
        raise KeyError(f"{variable} not in the dataset. "
                       f"Available: {list(ds.data_vars)}")

    default_label, default_cmap, default_sym = VARIABLES.get(
        variable, (variable, "bwr", False))
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
        ax.add_feature(cfeature.OCEAN, facecolor="#f3f3f3")
        ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                       edgecolor="#f0f2f4", linewidth=0.4)

        # pcolormesh, not imshow. imshow resamples whenever the image is
        # drawn smaller than its pixel count, and averaging neighbouring
        # cells pulls values toward the middle of the range, which on bwr
        # is white: the panels come out faded. pcolormesh draws each cell
        # as a filled polygon, with no averaging, so the colours hold.
        ax.pcolormesh(lon2d, lat2d, data[t], cmap=cm, norm=norm,
                      shading="auto", transform=ccrs.PlateCarree(),
                      zorder=5)

        ax.add_feature(cfeature.STATES, edgecolor="#272525",
                       linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#E7DFDF",
                       linewidth=0.7, zorder=6)
        ax.set_title(f"{year}", fontsize=13, fontweight="bold")

        # Distribution inset, lower right, rotated so its value axis runs
        # parallel with the colourbar and the two scales read together.
        v = data[t].ravel()
        v = v[np.isfinite(v)]
        if v.size:
            ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
            lim_lo, lim_hi = norm.vmin, norm.vmax
            ins.hist(np.clip(v, lim_lo, lim_hi), bins=30, color="#7dc98a",
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
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.9, aspect=30, pad=0.02,
                      extend="both")
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=9)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.06)

    _show(fig)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# All four livestock types
# ---------------------------------------------------------------------------
# Each archived file names its coefficient differently (dairy_Wccs_adjusted,
# beef_Wccs_adjusted, and so on), so the WCC variable is resolved per file
# rather than assumed. Everything else is passed straight to plot() above,
# so the formatting is identical to the single-type figures.

NETCDF_DIR = "/scratch/hdagne1/LivestockWaterUse/results/netCDF"

# File stem and display name. The stems are not uniform in the archive --
# hogs carries an extra "County" -- so they are listed, not constructed.
LIVESTOCK = [
    ("Dairy_Cattle", "Dairy cattle"),
    ("Beef_Cattle", "Beef cattle"),
    ("Hogs_County", "Hogs"),
    ("Poultry", "Poultry"),
]


def wcc_name(ds) -> str:
    """The per-type WCC variable, whatever this file calls it."""
    for v in ds.data_vars:
        if str(v).endswith("_Wccs_adjusted"):
            return str(v)
    raise KeyError(f"no *_Wccs_adjusted variable in {list(ds.data_vars)}")


def open_livestock(stem: str, netcdf_dir: str = NETCDF_DIR):
    """Open one archived livestock file."""
    return xr.open_dataset(
        f"{netcdf_dir}/County_Level_{stem}_WC_WW_1985_2022_geo_USGS_WC_WW.nc",
        engine="netcdf4")


def plot_livestock(variable: str = "WCC", netcdf_dir: str = NETCDF_DIR,
                   years: Optional[Sequence[int]] = None,
                   cmap: Optional[str] = None, clip_pct: float = 2.0,
                   ncol: int = 3, label: Optional[str] = None,
                   symmetric: Optional[bool] = None,
                   figsize: tuple = (14, 16), block_gap: float = 0.075):
    """
    ONE figure with four stacked blocks, one per livestock type, each a
    2 x 3 grid of benchmark years with its own colourbar.

    variable="WCC" resolves to each file's own coefficient name; anything
    else is passed through unchanged, so "CL_ratio", "Residual", "CL_WC"
    and "CL_WW" all work.

    Every block keeps its own colour scale. The four types differ by
    orders of magnitude for WCC and WC -- dairy runs some 150 times
    poultry -- so a single scale across the figure would render three of
    the four blocks a flat colour.

        plot_livestock("WCC")
        plot_livestock("CL_ratio")
        plot_livestock("Residual")
    """
    loaded = []
    for stem, name in LIVESTOCK:
        ds = open_livestock(stem, netcdf_dir)
        var = wcc_name(ds) if variable == "WCC" else variable
        if var not in ds:
            raise KeyError(f"{name}: {var} not in the file. "
                           f"Available: {list(ds.data_vars)}")

        available = ds.time.dt.year.values
        yrs = list(years or [y for y in (1985, 1990, 1995, 2000, 2010, 2015)
                             if y in available])
        missing = [y for y in yrs if y not in available]
        if missing:
            raise ValueError(f"{name}: years not in the file: {missing}")

        if var in NEEDS_POPULATION:
            bad = [y for y in yrs if y < min(CENSUS_YEARS)]
            if bad:
                raise ValueError(
                    f"{var} requires a livestock population, which the "
                    f"census provides from {min(CENSUS_YEARS)}. Requested "
                    f"years without one: {bad}.")

        idx = [int(np.where(available == y)[0][0]) for y in yrs]
        loaded.append((name, var, ds, ds[var].values[idx], yrs))

    default_label, default_cmap, default_sym = VARIABLES.get(
        loaded[0][1], (loaded[0][1], "bwr", False))
    label = label or default_label
    cm = plt.get_cmap(cmap or default_cmap)
    symmetric = default_sym if symmetric is None else symmetric

    yrs = loaded[0][4]
    nrow = int(np.ceil(len(yrs) / ncol))
    letters = "abcdefgh"

    fig = plt.figure(figsize=figsize, dpi=200)
    fig.subplots_adjust(wspace=0.02, hspace=0.07)
    top_margin = 0.955
    block_h = (top_margin - block_gap * (len(loaded) - 1)) / len(loaded)

    for bi, (name, var, ds, data, yrs) in enumerate(loaded):
        top = top_margin - bi * (block_h + block_gap)
        gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.88,
                              top=top, bottom=top - block_h,
                              wspace=0.02, hspace=0.07)

        norm = make_norm(data, symmetric, clip_pct)
        lon2d, lat2d = np.meshgrid(ds.lon.values, ds.lat.values)
        print(f"\n{name}  --  {var}")

        axes = []
        for k, year in enumerate(yrs):
            ax = fig.add_subplot(gs[k // ncol, k % ncol],
                                 projection=ccrs.LambertConformal())
            ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
            ax.set_aspect("auto")

            ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
            ax.add_feature(cfeature.OCEAN, facecolor="#f3f3f3")
            ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                           edgecolor="#f0f2f4", linewidth=0.4)

            ax.pcolormesh(lon2d, lat2d, data[k], cmap=cm, norm=norm,
                          shading="auto", transform=ccrs.PlateCarree(),
                          zorder=5)

            ax.add_feature(cfeature.STATES, edgecolor="#272525",
                           linewidth=0.5, zorder=6)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#E7DFDF",
                           linewidth=0.7, zorder=6)
            # ax.set_title(f"{year}", fontsize=13, fontweight="bold")
            ax.set_title(f"{year}", fontsize=13, fontweight="bold", y=1.02)

            v = data[k].ravel()
            v = v[np.isfinite(v)]
            if v.size:
                ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
                lim_lo, lim_hi = norm.vmin, norm.vmax
                ins.hist(np.clip(v, lim_lo, lim_hi), bins=30,
                         color="#7dc98a", edgecolor="black", linewidth=0.4,
                         orientation="horizontal")
                mean_val = float(np.mean(v))
                ins.axhline(np.clip(mean_val, lim_lo, lim_hi), color="red",
                            linestyle="--", linewidth=1.4)
                ins.text(1.0, 1.03, f"{mean_val:.4g}", color="blue",
                         ha="right", va="bottom", fontsize=8,
                         transform=ins.transAxes)
                ins.set_ylim(lim_lo, lim_hi)
                ins.invert_xaxis()
                ins.set_xticks([])
                ins.set_yticks([])
                ins.patch.set_alpha(0.85)

            print(f"  {year}  n={v.size:,}  mean {np.mean(v):+.4g}  "
                  f"median {np.median(v):+.4g}  "
                  f"range [{np.min(v):.4g}, {np.max(v):.4g}]")
            axes.append(ax)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, shrink=0.9, aspect=30, pad=0.02,
                          extend="both")
        cb.set_label(label, fontsize=12)
        cb.ax.tick_params(labelsize=9)

        # Above the panel titles, not beside them: at the same height the
        # block label runs into the first year and the two overlap.
        axes[0].text(-0.02, 1.42, f"{letters[bi]}) {name}",
                     transform=axes[0].transAxes, fontsize=12,
                     fontweight="bold", va="bottom", ha="left")

    _show(fig)
    plt.close(fig)
    return fig


def plot_wcc(**kw):
    """WCC, one figure per livestock type."""
    return plot_livestock("WCC", **kw)


def plot_ratio(**kw):
    """The downscaled WC-to-WW consumption-ratio, one figure per type."""
    return plot_livestock("CL_ratio", **kw)


def plot_residual(**kw):
    """The stored Residual field, one figure per type."""
    return plot_livestock("Residual", **kw)








# """
# plot_by_livestock.py

# Three figures from the archived netCDF, each showing one quantity for all
# four livestock types:

#     WCC        four blocks, one per livestock type
#     ratio      the downscaled WC-to-WW consumption-ratio
#     residual   the stored Residual field

# Every block is a 2 x 3 grid of the six benchmark years, with its own
# colourbar, matching the layout used elsewhere in the manuscript.

# The figures are DISPLAYED. Nothing is written to disk.

# Usage
# -----
#     from plot_by_livestock import plot_wcc, plot_ratio, plot_residual
#     plot_wcc()
#     plot_ratio()
#     plot_residual()
#     plot_wcc(figsize=(12, 14))
# """

# from __future__ import annotations

# import io
# from typing import Dict, List, Optional, Sequence

# import numpy as np

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# NETCDF_DIR = "/scratch/hdagne1/LivestockWaterUse/results/netCDF"
# CONUS_EXTENT = [-125, -66, 24, 50]
# YEARS = (1985, 1990, 1995, 2000, 2010, 2015)

# # File stem, block letter and display name per livestock type. The stems
# # are not uniform in the archive -- hogs carries an extra "County" -- so
# # they are listed rather than constructed.
# LIVESTOCK = [
#     ("Dairy_Cattle", "a", "Dairy cattle"),
#     ("Beef_Cattle", "b", "Beef cattle"),
#     ("Hogs_County", "c", "Hogs"),
#     ("Poultry", "d", "Poultry"),
# ]


# def _show(fig, dpi: int = 130) -> None:
#     try:
#         import IPython
#         ip = IPython.get_ipython()
#     except Exception:
#         ip = None
#     if ip is None:
#         plt.show()
#         return
#     from IPython.display import Image, display
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
#                 facecolor="white")
#     buf.seek(0)
#     display(Image(data=buf.getvalue()))
#     plt.close(fig)


# def wcc_name(ds) -> str:
#     """The per-type WCC variable, whatever this file calls it."""
#     for v in ds.data_vars:
#         if str(v).endswith("_Wccs_adjusted"):
#             return str(v)
#     raise KeyError(f"no *_Wccs_adjusted variable in {list(ds.data_vars)}")


# def field(ds, quantity: str) -> np.ndarray:
#     """
#     Pull one quantity from a dataset, resolving names that vary by file.

#     "residual" returns the stored Residual field. "wc_minus_ww" computes
#     the difference instead; note that WW is derived as WC divided by a
#     ratio bounded above by one, so WW is never smaller than WC and this
#     difference is negative wherever both are defined.
#     """
#     if quantity == "wcc":
#         return ds[wcc_name(ds)].values
#     if quantity == "ratio":
#         return ds["CL_ratio"].values
#     if quantity == "residual":
#         return ds["Residual"].values
#     if quantity == "wc_minus_ww":
#         return ds["CL_WC"].values - ds["CL_WW"].values
#     raise KeyError(f"unknown quantity: {quantity}")


# def plot_quantity(quantity: str, label: str, netcdf_dir: str = NETCDF_DIR,
#                   years: Sequence[int] = YEARS, cmap: str = "bwr",
#                   figsize: tuple = (14, 16),
#                   ncol: int = 3, wspace: float = 0.02, hspace: float = 0.07,
#                   block_gap: float = 0.075, shared: bool = False,
#                   title: Optional[str] = None):
#     """
#     One figure, four blocks, one block per livestock type.

#     figsize sets the figure directly. The maps use set_aspect("auto") and
#     stretch to fill whatever box they are given, so changing it alters
#     only the figure's width and height and no gaps open between columns.

#     shared=False gives each livestock type its own colourbar, which is the
#     default because the four differ by orders of magnitude for WCC; the
#     ratio is dimensionless and comparable, so shared=True suits it.
#     """
#     import xarray as xr

#     nrow = int(np.ceil(len(years) / ncol))
#     cm = plt.get_cmap(cmap)

#     loaded = []
#     for stem, letter, name in LIVESTOCK:
#         path = (f"{netcdf_dir}/County_Level_{stem}"
#                 f"_WC_WW_1985_2022_geo_USGS_WC_WW.nc")
#         ds = xr.open_dataset(path)
#         available = ds.time.dt.year.values
#         missing = [y for y in years if y not in available]
#         if missing:
#             raise ValueError(f"{name}: years not in the file: {missing}")
#         idx = [int(np.where(available == y)[0][0]) for y in years]
#         loaded.append((letter, name, ds, field(ds, quantity)[idx]))

#     if shared:
#         allv = np.concatenate([d.ravel() for _, _, _, d in loaded])
#         allv = allv[np.isfinite(allv)]
#         gmin, gmax = float(allv.min()), float(allv.max())

#     ds0 = loaded[0][2]
#     lat, lon = ds0.lat.values, ds0.lon.values
#     # pcolormesh, not imshow. imshow resamples whenever the image is drawn
#     # smaller than its pixel count, and averaging neighbouring cells pulls
#     # values toward the middle of the range -- which on bwr is white, so
#     # the panels come out faded. pcolormesh draws each cell as a filled
#     # polygon, with no averaging, and the colours stay saturated.
#     lon2d, lat2d = np.meshgrid(lon, lat)

#     fig = plt.figure(figsize=figsize, dpi=200)
#     fig.subplots_adjust(wspace=0.02, hspace=0.07)
#     top_margin = 0.955
#     block_h = (top_margin - block_gap * (len(loaded) - 1)) / len(loaded)

#     for bi, (letter, name, ds, data) in enumerate(loaded):
#         top = top_margin - bi * (block_h + block_gap)
#         gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.88,
#                               top=top, bottom=top - block_h,
#                               wspace=0.02, hspace=0.07)

#         v = data[np.isfinite(data)]
#         vmin, vmax = ((gmin, gmax) if shared
#                       else (float(v.min()), float(v.max())))

#         axes = []
#         for k, year in enumerate(years):
#             ax = fig.add_subplot(gs[k // ncol, k % ncol],
#                                  projection=ccrs.LambertConformal())
#             ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
#             ax.set_aspect("auto")

#             ax.add_feature(cfeature.LAND, facecolor="white")
#             ax.add_feature(cfeature.OCEAN, facecolor="white")
#             ax.add_feature(cfeature.LAKES, facecolor="white",
#                            edgecolor="#c8d4dc", linewidth=0.3)

#             im = ax.pcolormesh(lon2d, lat2d, data[k], cmap=cm,
#                                vmin=vmin, vmax=vmax, shading="auto",
#                                transform=ccrs.PlateCarree(), zorder=5)

#             ax.add_feature(cfeature.STATES, edgecolor="#FBF7F7",
#                            linewidth=0.35, zorder=6)
#             ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
#                            linewidth=0.5, zorder=6)
#             ax.set_title(f"{year}", fontsize=9)

#             # Distribution inset, lower right, rotated so its value axis
#             # runs parallel with the colourbar.
#             yv = data[k].ravel()
#             yv = yv[np.isfinite(yv)]
#             if yv.size:
#                 ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
#                 ins.hist(np.clip(yv, vmin, vmax), bins=25, color="#c1c97d",
#                          edgecolor="black", linewidth=0.3,
#                          orientation="horizontal")
#                 mean_val = float(np.mean(yv))
#                 ins.axhline(np.clip(mean_val, vmin, vmax), color="red",
#                             linestyle="--", linewidth=1.2)
#                 ins.text(1.0, 1.03, f"{mean_val:.4g}", color="blue",
#                          ha="right", va="bottom", fontsize=7,
#                          transform=ins.transAxes)
#                 ins.set_ylim(vmin, vmax)
#                 ins.invert_xaxis()
#                 ins.set_xticks([])
#                 ins.set_yticks([])
#                 ins.patch.set_alpha(0.85)
#             axes.append(ax)

#         cb = fig.colorbar(im, ax=axes, shrink=0.95, aspect=26, pad=0.015,
#                           fraction=0.03)
#         cb.set_label(label, fontsize=8)
#         cb.ax.tick_params(labelsize=7)

#         # Above the panel title, not beside it: at the same height the
#         # label runs into the year and the two overlap.
#         axes[0].text(-0.02, 1.42, f"{letter}) {name}",
#                      transform=axes[0].transAxes, fontsize=11,
#                      fontweight="bold", va="bottom", ha="left")

#         print(f"  {letter}) {name:<14} range [{v.min():.4g}, {v.max():.4g}]  "
#               f"mean {v.mean():.4g}")

#     if title:
#         fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
#     _show(fig)
#     plt.close(fig)
#     return fig


# def plot_wcc(**kw):
#     """WCC for all four livestock types."""
#     kw.setdefault("label", "WCC [gal head$^{-1}$ day$^{-1}$]")
#     return plot_quantity("wcc", **kw)


# def plot_ratio(**kw):
#     """The downscaled WC-to-WW consumption-ratio, shared scale."""
#     kw.setdefault("label", "Ratio [-]")
#     kw.setdefault("shared", True)
#     return plot_quantity("ratio", **kw)


# def plot_residual(quantity: str = "residual", **kw):
#     """
#     The residual field. quantity="wc_minus_ww" computes WC - WW instead of
#     reading the stored Residual.
#     """
#     kw.setdefault("label", "Residual [Mgal day$^{-1}$]")
#     return plot_quantity(quantity, **kw)






# """
# plot_by_livestock.py

# Three figures from the archived netCDF, each showing one quantity for all
# four livestock types:

#     WCC        four blocks, one per livestock type
#     ratio      the downscaled WC-to-WW consumption-ratio
#     residual   the stored Residual field

# Every block is a 2 x 3 grid of the six benchmark years, with its own
# colourbar, matching the layout used elsewhere in the manuscript.

# The figures are DISPLAYED. Nothing is written to disk.

# Usage
# -----
#     from plot_by_livestock import plot_wcc, plot_ratio, plot_residual
#     plot_wcc()
#     plot_ratio()
#     plot_residual()
#     plot_wcc(figsize=(12, 14))
# """

# from __future__ import annotations

# import io
# from typing import Dict, List, Optional, Sequence

# import numpy as np

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# NETCDF_DIR = "/scratch/hdagne1/LivestockWaterUse/results/netCDF"
# CONUS_EXTENT = [-125, -66, 24, 50]
# YEARS = (1985, 1990, 1995, 2000, 2010, 2015)

# # File stem, block letter and display name per livestock type. The stems
# # are not uniform in the archive -- hogs carries an extra "County" -- so
# # they are listed rather than constructed.
# LIVESTOCK = [
#     ("Dairy_Cattle", "a", "Dairy cattle"),
#     ("Beef_Cattle", "b", "Beef cattle"),
#     ("Hogs_County", "c", "Hogs"),
#     ("Poultry", "d", "Poultry"),
# ]


# def _show(fig, dpi: int = 130) -> None:
#     try:
#         import IPython
#         ip = IPython.get_ipython()
#     except Exception:
#         ip = None
#     if ip is None:
#         plt.show()
#         return
#     from IPython.display import Image, display
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
#                 facecolor="white")
#     buf.seek(0)
#     display(Image(data=buf.getvalue()))
#     plt.close(fig)


# def wcc_name(ds) -> str:
#     """The per-type WCC variable, whatever this file calls it."""
#     for v in ds.data_vars:
#         if str(v).endswith("_Wccs_adjusted"):
#             return str(v)
#     raise KeyError(f"no *_Wccs_adjusted variable in {list(ds.data_vars)}")


# def field(ds, quantity: str) -> np.ndarray:
#     """
#     Pull one quantity from a dataset, resolving names that vary by file.

#     "residual" returns the stored Residual field. "wc_minus_ww" computes
#     the difference instead; note that WW is derived as WC divided by a
#     ratio bounded above by one, so WW is never smaller than WC and this
#     difference is negative wherever both are defined.
#     """
#     if quantity == "wcc":
#         return ds[wcc_name(ds)].values
#     if quantity == "ratio":
#         return ds["CL_ratio"].values
#     if quantity == "residual":
#         return ds["Residual"].values
#     if quantity == "wc_minus_ww":
#         return ds["CL_WC"].values - ds["CL_WW"].values
#     raise KeyError(f"unknown quantity: {quantity}")


# def plot_quantity(quantity: str, label: str, netcdf_dir: str = NETCDF_DIR,
#                   years: Sequence[int] = YEARS, cmap: str = "jet",
#                   figsize: tuple = (14, 16),
#                   ncol: int = 3, wspace: float = 0.02, hspace: float = 0.25,
#                   block_gap: float = 0.075, shared: bool = False,
#                   title: Optional[str] = None):
#     """
#     One figure, four blocks, one block per livestock type.

#     figsize sets the figure directly. The maps use set_aspect("auto") and
#     stretch to fill whatever box they are given, so changing it alters
#     only the figure's width and height and no gaps open between columns.

#     shared=False gives each livestock type its own colourbar, which is the
#     default because the four differ by orders of magnitude for WCC; the
#     ratio is dimensionless and comparable, so shared=True suits it.
#     """
#     import xarray as xr

#     nrow = int(np.ceil(len(years) / ncol))
#     cm = plt.get_cmap(cmap)

#     loaded = []
#     for stem, letter, name in LIVESTOCK:
#         path = (f"{netcdf_dir}/County_Level_{stem}"
#                 f"_WC_WW_1985_2022_geo_USGS_WC_WW.nc")
#         ds = xr.open_dataset(path)
#         available = ds.time.dt.year.values
#         missing = [y for y in years if y not in available]
#         if missing:
#             raise ValueError(f"{name}: years not in the file: {missing}")
#         idx = [int(np.where(available == y)[0][0]) for y in years]
#         loaded.append((letter, name, ds, field(ds, quantity)[idx]))

#     if shared:
#         allv = np.concatenate([d.ravel() for _, _, _, d in loaded])
#         allv = allv[np.isfinite(allv)]
#         gmin, gmax = float(allv.min()), float(allv.max())

#     ds0 = loaded[0][2]
#     lat, lon = ds0.lat.values, ds0.lon.values
#     # pcolormesh, not imshow. imshow resamples whenever the image is drawn
#     # smaller than its pixel count, and averaging neighbouring cells pulls
#     # values toward the middle of the range -- which on bwr is white, so
#     # the panels come out faded. pcolormesh draws each cell as a filled
#     # polygon, with no averaging, and the colours stay saturated.
#     lon2d, lat2d = np.meshgrid(lon, lat)

#     fig = plt.figure(figsize=figsize, dpi=200)
#     fig.subplots_adjust(wspace=wspace, hspace=hspace)
#     top_margin = 0.955
#     block_h = (top_margin - block_gap * (len(loaded) - 1)) / len(loaded)

#     for bi, (letter, name, ds, data) in enumerate(loaded):
#         top = top_margin - bi * (block_h + block_gap)
#         gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.88,
#                               top=top, bottom=top - block_h,
#                               wspace=0.02, hspace=0.13)

#         v = data[np.isfinite(data)]
#         vmin, vmax = ((gmin, gmax) if shared
#                       else (float(v.min()), float(v.max())))

#         axes = []
#         for k, year in enumerate(years):
#             ax = fig.add_subplot(gs[k // ncol, k % ncol],
#                                  projection=ccrs.LambertConformal())
#             ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
#             ax.set_aspect("auto")

#             ax.add_feature(cfeature.LAND, facecolor="white")
#             ax.add_feature(cfeature.OCEAN, facecolor="white")
#             ax.add_feature(cfeature.LAKES, facecolor="white",
#                            edgecolor="#c8d4dc", linewidth=0.3)

#             im = ax.pcolormesh(lon2d, lat2d, data[k], cmap=cm,
#                                vmin=vmin, vmax=vmax, shading="auto",
#                                transform=ccrs.PlateCarree(), zorder=5)

#             ax.add_feature(cfeature.STATES, edgecolor="#888888",
#                            linewidth=0.35, zorder=6)
#             ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
#                            linewidth=0.5, zorder=6)
#             ax.set_title(f"{year}", fontsize=9)

#             # Distribution inset, lower right, rotated so its value axis
#             # runs parallel with the colourbar.
#             yv = data[k].ravel()
#             yv = yv[np.isfinite(yv)]
#             if yv.size:
#                 ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
#                 ins.hist(np.clip(yv, vmin, vmax), bins=25, color="#c1c97d",
#                          edgecolor="black", linewidth=0.3,
#                          orientation="horizontal")
#                 mean_val = float(np.mean(yv))
#                 ins.axhline(np.clip(mean_val, vmin, vmax), color="red",
#                             linestyle="--", linewidth=1.2)
#                 ins.text(1.0, 1.03, f"{mean_val:.4g}", color="blue",
#                          ha="right", va="bottom", fontsize=7,
#                          transform=ins.transAxes)
#                 ins.set_ylim(vmin, vmax)
#                 ins.invert_xaxis()
#                 ins.set_xticks([])
#                 ins.set_yticks([])
#                 ins.patch.set_alpha(0.85)
#             axes.append(ax)

#         cb = fig.colorbar(im, ax=axes, shrink=0.95, aspect=26, pad=0.015,
#                           fraction=0.03)
#         cb.set_label(label, fontsize=8)
#         cb.ax.tick_params(labelsize=7)

#         # Above the panel title, not beside it: at the same height the
#         # label runs into the year and the two overlap.
#         axes[0].text(-0.02, 1.42, f"{letter}) {name}",
#                      transform=axes[0].transAxes, fontsize=11,
#                      fontweight="bold", va="bottom", ha="left")

#         print(f"  {letter}) {name:<14} range [{v.min():.4g}, {v.max():.4g}]  "
#               f"mean {v.mean():.4g}")

#     if title:
#         fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
#     _show(fig)
#     plt.close(fig)
#     return fig


# def plot_wcc(**kw):
#     """WCC for all four livestock types."""
#     kw.setdefault("label", "WCC [gal head$^{-1}$ day$^{-1}$]")
#     return plot_quantity("wcc", **kw)


# def plot_ratio(**kw):
#     """The downscaled WC-to-WW consumption-ratio, shared scale."""
#     kw.setdefault("label", "Ratio [-]")
#     kw.setdefault("shared", True)
#     return plot_quantity("ratio", **kw)


# def plot_residual(quantity: str = "residual", **kw):
#     """
#     The residual field. quantity="wc_minus_ww" computes WC - WW instead of
#     reading the stored Residual.
#     """
#     kw.setdefault("label", "Residual [Mgal day$^{-1}$]")
#     return plot_quantity(quantity, **kw)




# """
# plot_by_livestock.py

# Three figures from the archived netCDF, each showing one quantity for all
# four livestock types:

#     WCC        four blocks, one per livestock type
#     ratio      the downscaled WC-to-WW consumption-ratio
#     residual   the stored Residual field

# Every block is a 2 x 3 grid of the six benchmark years, with its own
# colourbar, matching the layout used elsewhere in the manuscript.

# The figures are DISPLAYED. Nothing is written to disk.

# Usage
# -----
#     from plot_by_livestock import plot_wcc, plot_ratio, plot_residual
#     plot_wcc()
#     plot_ratio()
#     plot_residual()
#     plot_wcc(figsize=(12, 14))
# """

# from __future__ import annotations

# import io
# from typing import Dict, List, Optional, Sequence

# import numpy as np

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# NETCDF_DIR = "/scratch/hdagne1/LivestockWaterUse/results/netCDF"
# CONUS_EXTENT = [-125, -66, 24, 50]
# YEARS = (1985, 1990, 1995, 2000, 2010, 2015)

# # File stem, block letter and display name per livestock type. The stems
# # are not uniform in the archive -- hogs carries an extra "County" -- so
# # they are listed rather than constructed.
# LIVESTOCK = [
#     ("Dairy_Cattle", "a", "Dairy cattle"),
#     ("Beef_Cattle", "b", "Beef cattle"),
#     ("Hogs_County", "c", "Hogs"),
#     ("Poultry", "d", "Poultry"),
# ]


# def _show(fig, dpi: int = 130) -> None:
#     try:
#         import IPython
#         ip = IPython.get_ipython()
#     except Exception:
#         ip = None
#     if ip is None:
#         plt.show()
#         return
#     from IPython.display import Image, display
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
#                 facecolor="white")
#     buf.seek(0)
#     display(Image(data=buf.getvalue()))
#     plt.close(fig)


# def wcc_name(ds) -> str:
#     """The per-type WCC variable, whatever this file calls it."""
#     for v in ds.data_vars:
#         if str(v).endswith("_Wccs_adjusted"):
#             return str(v)
#     raise KeyError(f"no *_Wccs_adjusted variable in {list(ds.data_vars)}")


# def field(ds, quantity: str) -> np.ndarray:
#     """
#     Pull one quantity from a dataset, resolving names that vary by file.

#     "residual" returns the stored Residual field. "wc_minus_ww" computes
#     the difference instead; note that WW is derived as WC divided by a
#     ratio bounded above by one, so WW is never smaller than WC and this
#     difference is negative wherever both are defined.
#     """
#     if quantity == "wcc":
#         return ds[wcc_name(ds)].values
#     if quantity == "ratio":
#         return ds["CL_ratio"].values
#     if quantity == "residual":
#         return ds["Residual"].values
#     if quantity == "wc_minus_ww":
#         return ds["CL_WC"].values - ds["CL_WW"].values
#     raise KeyError(f"unknown quantity: {quantity}")


# def plot_quantity(quantity: str, label: str, netcdf_dir: str = NETCDF_DIR,
#                   years: Sequence[int] = YEARS, cmap: str = "bwr",
#                   figsize: tuple = (14, 16),
#                   ncol: int = 3, wspace: float = 0.02, hspace: float = 0.25,
#                   block_gap: float = 0.075, shared: bool = False,
#                   title: Optional[str] = None):
#     """
#     One figure, four blocks, one block per livestock type.

#     figsize sets the figure directly. The maps use set_aspect("auto") and
#     stretch to fill whatever box they are given, so changing it alters
#     only the figure's width and height and no gaps open between columns.

#     shared=False gives each livestock type its own colourbar, which is the
#     default because the four differ by orders of magnitude for WCC; the
#     ratio is dimensionless and comparable, so shared=True suits it.
#     """
#     import xarray as xr

#     nrow = int(np.ceil(len(years) / ncol))
#     cm = plt.get_cmap(cmap)

#     loaded = []
#     for stem, letter, name in LIVESTOCK:
#         path = (f"{netcdf_dir}/County_Level_{stem}"
#                 f"_WC_WW_1985_2022_geo_USGS_WC_WW.nc")
#         ds = xr.open_dataset(path)
#         available = ds.time.dt.year.values
#         missing = [y for y in years if y not in available]
#         if missing:
#             raise ValueError(f"{name}: years not in the file: {missing}")
#         idx = [int(np.where(available == y)[0][0]) for y in years]
#         loaded.append((letter, name, ds, field(ds, quantity)[idx]))

#     if shared:
#         allv = np.concatenate([d.ravel() for _, _, _, d in loaded])
#         allv = allv[np.isfinite(allv)]
#         gmin, gmax = float(allv.min()), float(allv.max())

#     ds0 = loaded[0][2]
#     lat, lon = ds0.lat.values, ds0.lon.values
#     # imshow, not pcolormesh: the grid is regular, and pcolormesh would
#     # draw millions of quadrilaterals per panel.
#     extent = [lon.min(), lon.max(), lat.min(), lat.max()]
#     origin = "upper" if lat[0] > lat[-1] else "lower"

#     fig = plt.figure(figsize=figsize, dpi=200)
#     fig.subplots_adjust(wspace=wspace, hspace=hspace)
#     top_margin = 0.955
#     block_h = (top_margin - block_gap * (len(loaded) - 1)) / len(loaded)

#     for bi, (letter, name, ds, data) in enumerate(loaded):
#         top = top_margin - bi * (block_h + block_gap)
#         gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.88,
#                               top=top, bottom=top - block_h,
#                               wspace=0.02, hspace=0.15)

#         v = data[np.isfinite(data)]
#         vmin, vmax = ((gmin, gmax) if shared
#                       else (float(v.min()), float(v.max())))

#         axes = []
#         for k, year in enumerate(years):
#             ax = fig.add_subplot(gs[k // ncol, k % ncol],
#                                  projection=ccrs.LambertConformal())
#             ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
#             ax.set_aspect("auto")

#             ax.add_feature(cfeature.LAND, facecolor="white")
#             ax.add_feature(cfeature.OCEAN, facecolor="white")
#             ax.add_feature(cfeature.LAKES, facecolor="white",
#                            edgecolor="#c8d4dc", linewidth=0.3)
#             lon2d, lat2d = np.meshgrid(lon, lat)

#             im = ax.pcolormesh(lon2d, lat2d, data[k], cmap=cm,
#                             vmin=vmin, vmax=vmax, shading="auto",
#                             transform=ccrs.PlateCarree(), zorder=5)

#             ax.add_feature(cfeature.STATES, edgecolor="#888888",
#                            linewidth=0.35, zorder=6)
#             ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
#                            linewidth=0.5, zorder=6)
#             ax.set_title(f"{year}", fontsize=9)

#             # Distribution inset, lower right, rotated so its value axis
#             # runs parallel with the colourbar.
#             yv = data[k].ravel()
#             yv = yv[np.isfinite(yv)]
#             if yv.size:
#                 ins = ax.inset_axes([0.78, 0.05, 0.19, 0.40])
#                 ins.hist(np.clip(yv, vmin, vmax), bins=25, color="#c1c97d",
#                          edgecolor="black", linewidth=0.3,
#                          orientation="horizontal")
#                 mean_val = float(np.mean(yv))
#                 ins.axhline(np.clip(mean_val, vmin, vmax), color="red",
#                             linestyle="--", linewidth=1.2)
#                 ins.text(1.0, 1.03, f"{mean_val:.4g}", color="blue",
#                          ha="right", va="bottom", fontsize=7,
#                          transform=ins.transAxes)
#                 ins.set_ylim(vmin, vmax)
#                 ins.invert_xaxis()
#                 ins.set_xticks([])
#                 ins.set_yticks([])
#                 ins.patch.set_alpha(0.85)
#             axes.append(ax)

#         cb = fig.colorbar(im, ax=axes, shrink=0.95, aspect=26, pad=0.015,
#                           fraction=0.03)
#         cb.set_label(label, fontsize=8)
#         cb.ax.tick_params(labelsize=7)

#         # Above the panel title, not beside it: at the same height the
#         # label runs into the year and the two overlap.
#         axes[0].text(-0.02, 1.42, f"{letter}) {name}",
#                      transform=axes[0].transAxes, fontsize=11,
#                      fontweight="bold", va="bottom", ha="left")

#         print(f"  {letter}) {name:<14} range [{v.min():.4g}, {v.max():.4g}]  "
#               f"mean {v.mean():.4g}")

#     if title:
#         fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
#     _show(fig)
#     plt.close(fig)
#     return fig


# def plot_wcc(**kw):
#     """WCC for all four livestock types."""
#     kw.setdefault("label", "WCC [gal head$^{-1}$ day$^{-1}$]")
#     return plot_quantity("wcc", **kw)


# def plot_ratio(**kw):
#     """The downscaled WC-to-WW consumption-ratio, shared scale."""
#     kw.setdefault("label", "Ratio [-]")
#     kw.setdefault("shared", True)
#     return plot_quantity("ratio", **kw)


# def plot_residual(quantity: str = "residual", **kw):
#     """
#     The residual field. quantity="wc_minus_ww" computes WC - WW instead of
#     reading the stored Residual.
#     """
#     kw.setdefault("label", "Residual [Mgal day$^{-1}$]")
#     return plot_quantity(quantity, **kw)