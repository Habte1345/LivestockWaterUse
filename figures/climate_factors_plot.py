"""
plot_climate_maps.py

County choropleth of the three core forcings -- precipitation, temperature
and relative humidity -- as a 1 x 3 panel over CONUS, on a cartopy Lambert
Conformal basemap with discrete, binned colourbars.

The figure is DISPLAYED. Nothing is written to disk.

Values are the long-term climatology over whatever span the tables cover:

    precipitation   annual total, averaged across years. Monthly rows are
                    summed within each year FIRST, then averaged -- taking
                    the mean of the monthly values directly would give a
                    mean month, twelve times too small.
    temperature     mean of all months
    relative hum.   mean of all months

County geometry comes from the Census cartographic boundary file, joined on
the 5-character FIPS every table in this project shares. Geometry is read
into memory; nothing is cached to the data directory.

Usage
-----
    python plot_climate_maps.py
    python plot_climate_maps.py --year 2015 --bins 8
    python plot_climate_maps.py --shapefile /path/to/cb_2018_us_county_5m.shp

    from plot_climate_maps import plot
    plot()
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/climate"

COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
              "cb_2018_us_county_5m.zip")

NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_EXTENT = [-125, -66, 24, 50]

# (file stem, value column, aggregation within a year, label)
PANELS = [
    ("Precp", "precip_mm_county", "sum", "Precipitation [mm yr$^{-1}$]"),
    ("Temp", "temp_c_county", "mean", "Temperature [$^{\\circ}$C]"),
    ("RH", "rh_pct_county", "mean", "Relative humidity [%]"),
]

CMAP = "bwr"
N_BINS = 6

# Trim the scale at these percentiles before binning. Using the raw min and
# max would let a handful of extreme counties absorb whole classes and
# leave the rest of the map in one or two colours.
CLIP_PCT = 2.0


def make_boundary_norm(values, cmap, n_bins: int = N_BINS,
                       clip_pct: float = CLIP_PCT):
    """
    Discrete colour classes over the trimmed data range.

    A BoundaryNorm rather than a continuous ramp: with a fixed number of
    equal-width classes each colour maps to a stated interval, so a county
    can be read off as a number instead of guessed at along a gradient.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return mcolors.BoundaryNorm(np.linspace(0, 1, n_bins + 1), cmap.N)

    vmin, vmax = np.nanpercentile(v, [clip_pct, 100 - clip_pct])
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else abs(vmin) * 0.05
        vmin, vmax = vmin - pad, vmax + pad
    edges = np.linspace(vmin, vmax, n_bins + 1)
    return mcolors.BoundaryNorm(edges, cmap.N)


def _show(fig, screen_dpi: int = 130) -> None:
    """
    Render whatever the kernel is configured to do.

    plt.show() is a silent no-op when the active backend is
    non-interactive, and display(fig) alone is no better, because
    matplotlib's PNG formatter is only registered by the inline backend.
    Encoding the PNG here and publishing the bytes depends on neither.
    """
    ip = None
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
    fig.savefig(buf, format="png", dpi=screen_dpi,
                bbox_inches="tight", facecolor="white")
    buf.seek(0)
    display(Image(data=buf.getvalue()))
    plt.close(fig)


def find_table(base: Path, stem: str) -> Path:
    hits = sorted(base.glob(f"{stem}_county_level_*.feather"))
    if not hits:
        raise FileNotFoundError(
            f"no {stem}_county_level_*.feather in {base}. "
            f"Run era5_forcings.py first.")
    return hits[-1]


def climatology(path: Path, col: str, how: str,
                year: Optional[int]) -> pd.DataFrame:
    """One value per county. Precipitation sums within a year, then averages."""
    df = pd.read_feather(path)
    if year is not None:
        df = df[df["year"] == year]
        if df.empty:
            raise ValueError(f"{path.name} has no rows for year {year}")

    df["fips"] = df["fips"].astype(str).str.zfill(5)

    if "month" in df.columns and how == "sum":
        per_year = df.groupby(["fips", "year"], as_index=False)[col].sum()
        return per_year.groupby("fips", as_index=False)[col].mean()
    return df.groupby("fips", as_index=False)[col].mean()


def load_counties(shapefile: Optional[str]):
    import geopandas as gpd
    g = gpd.read_file(shapefile if shapefile else COUNTY_URL)
    g = g[~g["STATEFP"].isin(NON_CONUS_FIPS)].copy()
    g["fips"] = g["GEOID"].astype(str).str.zfill(5)
    return g.to_crs("EPSG:4326")


def plot(data_dir: str = DATA_DIR, year: Optional[int] = None,
         shapefile: Optional[str] = None, cmap: str = CMAP,
         bins: int = N_BINS, clip_pct: float = CLIP_PCT):
    """Draw the 1 x 3 map and return the figure. Nothing is saved."""
    base = Path(data_dir).expanduser()
    counties = load_counties(shapefile)
    cm = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(19, 4.6), dpi=200)

    for i, (stem, col, how, label) in enumerate(PANELS, start=1):
        ax = fig.add_subplot(1, 3, i, projection=ccrs.LambertConformal())
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        ax.set_aspect("auto")

        ax.add_feature(cfeature.LAND, facecolor="#faf8f2")
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
        ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                       edgecolor="#8fa8b8", linewidth=0.4)

        vals = climatology(find_table(base, stem), col, how, year)
        g = counties.merge(vals, on="fips", how="left")

        v = g[col]
        n_missing = int(v.isna().sum())
        norm = make_boundary_norm(v.values, cm, bins, clip_pct)

        # Counties sit beneath the political and hydrographic linework so
        # the boundaries stay legible over the fill.
        g.plot(column=col, cmap=cm, norm=norm, ax=ax, linewidth=0.0,
               transform=ccrs.PlateCarree(), zorder=3,
               missing_kwds={"color": "0.85", "edgecolor": "0.6",
                             "linewidth": 0.1, "hatch": "///"})

        ax.add_feature(cfeature.STATES, edgecolor="#888888",
                       linewidth=0.5, zorder=4)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
                       linewidth=0.7, zorder=4)
        ax.add_feature(cfeature.RIVERS, edgecolor="#7fb3d5",
                       linewidth=0.4, alpha=0.5, zorder=5)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.96, pad=0.02, extend="both")
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=9)

        ax.set_title(f"{label.split(' [')[0]} (n={int(v.notna().sum()):,})",
                     fontsize=13, fontweight="bold")

        edges = norm.boundaries
        span = "climatology" if year is None else str(year)
        print(f"  {stem:<7} {span:<12} n={int(v.notna().sum()):,}  "
              f"min {np.nanmin(v):8.2f}  median {np.nanmedian(v):8.2f}  "
              f"max {np.nanmax(v):8.2f}  |  {bins} bins "
              f"[{edges[0]:.1f} .. {edges[-1]:.1f}]"
              + (f"   MISSING {n_missing}" if n_missing else ""))

    plt.tight_layout()
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="County maps of precipitation, temperature and RH.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--year", type=int, default=None,
                    help="single year; omit for the long-term mean")
    ap.add_argument("--shapefile", default=None,
                    help="local county shapefile; omit to read from Census")
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--bins", type=int, default=N_BINS,
                    help="number of discrete colour classes")
    ap.add_argument("--clip-pct", type=float, default=CLIP_PCT,
                    help="percentile trim before binning")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.year, args.shapefile,
         args.cmap, args.bins, args.clip_pct)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()