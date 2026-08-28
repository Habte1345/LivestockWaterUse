"""
plot_wc_maps.py

County-level water consumption (WC) for each livestock type, as a 2 x 2
panel over CONUS. The figure is DISPLAYED; nothing is written to disk.

Reads county_level_WC_<type>.feather from county_water_consumption.py.
Rows are county-month; WC is a rate (Mgal/d), so months average rather
than sum.

The colour scale is LOGARITHMIC by default. WC is WCC times head count,
and head counts span five or six orders of magnitude across counties --
a handful of feedlot and broiler counties against thousands with a few
dozen animals. On a linear scale those few counties take the whole ramp
and everything else collapses to one colour. Pass --linear to override.

Usage
-----
    python plot_wc_maps.py
    python plot_wc_maps.py --year 2017 --units m3
    from plot_wc_maps import plot; plot()
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
DATA_DIR = f"{_REPO}/data/wc_county"

PANELS = [("dairy_cattle", "Dairy cattle"), ("beef_cattle", "Beef cattle"),
          ("hogs", "Hogs"), ("poultry", "Poultry")]

NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_EXTENT = [-125, -66, 24, 50]
COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
              "cb_2018_us_county_20m.zip")

CMAP, CLIP_PCT = "RdYlGn_r", 1.0
UNITS = {"mgal": ("wc_mgal_d", "WC [Mgal d$^{-1}$]"),
         "m3": ("wc_m3_d", "WC [m$^3$ d$^{-1}$]"),
         "l": ("wc_l_d", "WC [L d$^{-1}$]")}


def _show(fig, dpi: int = 130) -> None:
    """
    plt.show() is a silent no-op on a non-interactive backend, and
    display(fig) falls back to a text repr unless the inline backend
    registered its PNG formatter, so the bytes are encoded here instead.
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


def make_norm(values, log: bool, clip_pct: float):
    """
    Trimmed scale. On a log scale only strictly positive values can be
    shown, so zero-consumption counties are excluded from the range and
    drawn as missing rather than silently clipped to the floor.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        return mcolors.Normalize(0, 1)
    lo, hi = np.nanpercentile(v, [clip_pct, 100 - clip_pct])
    if lo >= hi:
        lo, hi = v.min(), max(v.max(), v.min() * 1.05 + 1e-12)
    return (mcolors.LogNorm(vmin=max(lo, 1e-12), vmax=hi) if log
            else mcolors.Normalize(vmin=lo, vmax=hi))


def county_mean(base: Path, stem: str, col: str,
                year: Optional[int]) -> pd.DataFrame:
    """One value per county. WC is a rate, so months average."""
    p = base / f"county_level_WC_{stem}.feather"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run county_water_consumption.py first.")
    df = pd.read_feather(p)
    if year is not None:
        df = df[df["year"] == year]
        if df.empty:
            raise ValueError(f"{stem}: no rows for {year}")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df.groupby("fips", as_index=False)[col].mean()


def load_counties(shapefile: Optional[str]):
    import geopandas as gpd
    g = gpd.read_file(shapefile if shapefile else COUNTY_URL)
    g = g[~g["STATEFP"].isin(NON_CONUS_FIPS)].copy()
    g["fips"] = g["GEOID"].astype(str).str.zfill(5)
    return g.to_crs("EPSG:4326")


def plot(data_dir: str = DATA_DIR, year: Optional[int] = None,
         shapefile: Optional[str] = None, cmap: str = CMAP,
         units: str = "mgal", log: bool = True, clip_pct: float = CLIP_PCT):
    """Draw the 2 x 2 map and return the figure. Nothing is saved."""
    base = Path(data_dir).expanduser()
    counties = load_counties(shapefile)
    cm = plt.get_cmap(cmap)
    col, label = UNITS[units]

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 13.5), dpi=200,
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)

    for ax, (stem, title) in zip(axes.ravel(), PANELS):
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

        vals = county_mean(base, stem, col, year)
        g = counties.merge(vals, on="fips", how="left")
        v = g[col]
        if log:                      # zeros cannot be shown on a log scale
            g.loc[v <= 0, col] = np.nan
            v = g[col]
        norm = make_norm(v.values, log, clip_pct)

        g.plot(column=col, cmap=cm, norm=norm, ax=ax, edgecolor="none",
               transform=ccrs.PlateCarree(), zorder=5,
               missing_kwds={"color": "white", "edgecolor": "none"})
        ax.add_feature(cfeature.STATES, edgecolor="gray",
                       linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=6)
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.82, aspect=22, pad=0.02,
                          extend="both")
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=8)

        n_ok, n_miss = int(v.notna().sum()), int(v.isna().sum())
        ax.set_title(f"{title} (n={n_ok:,})", fontsize=13,
                     fontweight="bold", loc="center")
        print(f"  {title:<14}{'all years' if year is None else year}  "
              f"n={n_ok:,}  min {np.nanmin(v):.4g}  "
              f"median {np.nanmedian(v):.4g}  max {np.nanmax(v):.4g}  |  "
              f"{'log' if log else 'linear'} scale "
              f"[{norm.vmin:.4g} .. {norm.vmax:.4g}]"
              + (f"   no value {n_miss:,}" if n_miss else ""))

    fig.suptitle("County-level livestock water consumption"
                 + (f", {year}" if year else ""),
                 fontsize=15, fontweight="bold")
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="County water consumption maps by livestock type.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--shapefile", default=None)
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--units", choices=tuple(UNITS), default="mgal")
    ap.add_argument("--linear", action="store_true",
                    help="linear colour scale; log is the default because "
                         "head counts span orders of magnitude")
    ap.add_argument("--clip-pct", type=float, default=CLIP_PCT)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.year, args.shapefile, args.cmap,
         args.units, not args.linear, args.clip_pct)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()