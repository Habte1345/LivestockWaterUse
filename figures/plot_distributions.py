"""
plot_distributions.py

Standardised distributions of the modelled and reference quantities: a
two-row figure with probability density above and cumulative distribution
below, one column per livestock type.

The figure is DISPLAYED. Nothing is written to disk.

Every series is converted to a z-score before plotting. The quantities
differ by orders of magnitude -- a population in head against a ratio near
one -- so they cannot share an axis in their native units. Standardising
puts them on a common scale, at the cost that the axis shows relative
position rather than magnitude: the comparison is of distributional SHAPE,
not of level.

The variable behind each series is set in SERIES and can be changed in one
place, since the archived files do not name every quantity obviously.

Usage
-----
    from plot_distributions import plot
    plot()
    plot(series={"USGS WC": ("VALUE", "#e6007e"),
                 "Pred WC": ("CL_WC", "#4d4d4d")})
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt

NETCDF_DIR = "/scratch/hdagne1/LivestockWaterUse/results/netCDF"

LIVESTOCK = [
    ("Dairy_Cattle", "a", "Dairy cattle"),
    ("Beef_Cattle", "b", "Beef cattle"),
    ("Hogs_County", "c", "Hogs"),
    ("Poultry", "d", "Poultry"),
]

# Label -> (variable, colour). "WCC" resolves to whichever
# *_Wccs_adjusted variable the file carries, since it is named per type.
#
# VERIFY THESE BEFORE PUBLISHING. The archive contains a variable called
# VALUE whose meaning is not evident from its name; it is mapped to the
# USGS reference on the assumption that it holds the observed water use,
# and that assumption should be checked against the source.
SERIES: Dict[str, tuple] = {
    "USGS WC": ("VALUE", "#e40c78"),
    "Predicted WC": ("CL_WC", "#060af7"),
    "Predicted WCC": ("WCC", "#0a0b0c"),
    "Ratio": ("CL_ratio", "#ece927"),
    "Residual": ("Residual", "#3df25b"),
}


def _show(fig, dpi: int = 130) -> None:
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


def wcc_name(ds) -> str:
    """The per-type WCC variable, whatever this file calls it."""
    for v in ds.data_vars:
        if str(v).endswith("_Wccs_adjusted"):
            return str(v)
    raise KeyError(f"no *_Wccs_adjusted variable in {list(ds.data_vars)}")


MAX_SAMPLE = 500_000


def zscore(v, rng=None) -> np.ndarray:
    """
    Standardise, reducing the array before any full-length operation.

    A county-year grid holds around two hundred million values per
    variable. Both the finite-value mask and the cumulative sort are
    O(n) with a copy, so touching the whole array twice costs minutes for
    a curve that half a million values already fix beyond plotting
    precision.

    The array is therefore strided down to roughly two million values
    first, and the mean and standard deviation are computed from that
    sample rather than from the full grid. On this many values those
    estimates agree with the exact ones to more digits than the figure
    can show, and a regular stride over a raster samples the domain
    evenly rather than favouring any region.
    """
    v = np.asarray(v).ravel()
    if v.size > 4 * MAX_SAMPLE:
        v = v[:: max(1, v.size // (4 * MAX_SAMPLE))]
    v = v.astype(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return v

    mean, sd = v.mean(), v.std()
    if v.size > MAX_SAMPLE:
        rng = rng or np.random.default_rng(0)
        v = v[rng.integers(0, v.size, MAX_SAMPLE)]
    return (v - mean) / sd if sd > 0 else v - mean


def density(z: np.ndarray, grid: np.ndarray, bins: int = 400) -> np.ndarray:
    """
    Smoothed density from a histogram rather than a kernel sum.

    A direct Gaussian kernel evaluates every grid point against every
    sample: at five million county-year values and 400 grid points that is
    two billion exponentials per series, which is what made an earlier
    version of this figure unusable. Binning first and smoothing the
    counts costs one pass over the data and is visually indistinguishable
    at this sample size.
    """
    lo, hi = grid[0], grid[-1]
    counts, edges = np.histogram(z, bins=bins, range=(lo, hi), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Gaussian smoothing over the bins, width set to a few bin widths.
    step = centres[1] - centres[0]
    half = max(1, int(round(0.06 / step)))
    k = np.exp(-0.5 * (np.arange(-3 * half, 3 * half + 1) / half) ** 2)
    k /= k.sum()
    smooth = np.convolve(counts, k, mode="same")
    return np.interp(grid, centres, smooth)


def plot(netcdf_dir: str = NETCDF_DIR, series: Optional[Dict] = None,
         years: Optional[Sequence[int]] = None,
         figsize: tuple = (16, 6), xlim: tuple = (-3, 3),
         fill_alpha: float = 0.35):
    """Draw the density and cumulative panels and return the figure."""
    import xarray as xr

    series = series or SERIES
    ncol = len(LIVESTOCK)
    grid = np.linspace(*xlim, 600)

    fig, axes = plt.subplots(2, ncol, figsize=figsize, dpi=200,
                             sharex=True, sharey="row")
    fig.subplots_adjust(wspace=0.10, hspace=0.12)

    for ci, (stem, letter, name) in enumerate(LIVESTOCK):
        ds = xr.open_dataset(
            f"{netcdf_dir}/County_Level_{stem}"
            f"_WC_WW_1985_2022_geo_USGS_WC_WW.nc", engine="netcdf4")
        if years is not None:
            ds = ds.sel(time=ds.time.dt.year.isin(list(years)))

        ax_p, ax_c = axes[0, ci], axes[1, ci]
        shown = []

        for label, (var, colour) in series.items():
            v = wcc_name(ds) if var == "WCC" else var
            if v not in ds:
                continue
            z = zscore(ds[v].values)
            if z.size < 10:
                continue

            dens = density(z, grid)
            ax_p.fill_between(grid, dens, color=colour, alpha=fill_alpha,
                              lw=0)
            ax_p.plot(grid, dens, color=colour, lw=1.4,
                      label=label if ci == 0 else None)

            # Empirical CDF from a coarse quantile grid: plotting five
            # million sorted points draws the same curve far more slowly.
            q = np.linspace(0, 1, 400)
            ax_c.plot(np.quantile(z, q), q, color=colour, lw=1.6)
            shown.append(label)

        ax_p.set_title(f"{letter}) {name}", fontsize=12, fontweight="bold")
        for ax in (ax_p, ax_c):
            ax.set_xlim(*xlim)
            ax.axvline(0, color="0.75", lw=0.7, ls=":")
            ax.tick_params(labelsize=9)
            ax.grid(alpha=0.18, lw=0.5)
            # for side in ("top", "right"):
            #     ax.spines[side].set_visible(False)
        ax_c.set_ylim(0, 1)

        if ci == 0:
            ax_p.set_ylabel("Density", fontsize=16)
            ax_c.set_ylabel("Cumulative probability", fontsize=16)
            ax_p.legend(fontsize=8, frameon=False, loc="upper right",
                        handlelength=1.4, labelspacing=0.3)

        print(f"  {letter}) {name:<14}{', '.join(shown)}")

    fig.supxlabel("Standardised value (z-score)", fontsize=11)
    _show(fig)
    plt.close(fig)
    return fig