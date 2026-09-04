"""
plot_wcc_maps.py

County-level water consumption coefficient for each livestock type, as a
2 x 2 panel over CONUS on a cartopy Lambert Conformal basemap.

The figure is DISPLAYED. Nothing is written to disk.

Reads wcc_county_<type>_<start>_<end>.feather from wcc_ann_downscale.py.
Rows are county-month, so they are averaged to one value per county:
a WCC is a rate, not a total, so it averages rather than sums.

Each panel is scaled independently. A dairy cow drinks roughly 500 times
what a chicken does, so a shared colour scale would render three of the
four panels a single flat colour.

Usage
-----
    python plot_wcc_maps.py
    python plot_wcc_maps.py --year 2015 --units gal
    from plot_wcc_maps import plot; plot()
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
DATA_DIR = f"{_REPO}/data/wcc_county"

PANELS = [("dairy_cattle", "Dairy cattle"), ("beef_cattle", "Beef cattle"),
          ("hogs", "Hogs"), ("poultry", "Poultry")]

NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_EXTENT = [-125, -66, 24, 50]
COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
              "cb_2018_us_county_5m.zip")

CMAP, CLIP_PCT = "bwr", 2.0
GAL_PER_L = 1.0 / 3.785411784


def make_norm(values, clip_pct=CLIP_PCT):
    """
    Continuous linear scale over the trimmed data range.

    Still trimmed at the 2nd and 98th percentile: with raw min and max a
    couple of extreme counties would compress everything else into a
    narrow slice of the ramp.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return mcolors.Normalize(0, 1)
    lo, hi = np.nanpercentile(v, [clip_pct, 100 - clip_pct])
    if lo == hi:
        pad = 0.5 if lo == 0 else abs(lo) * 0.05
        lo, hi = lo - pad, hi + pad
    return mcolors.Normalize(vmin=lo, vmax=hi)


def _show(fig, dpi: int = 130) -> None:
    """
    Render whatever the kernel is configured to do. plt.show() is a silent
    no-op on a non-interactive backend, and display(fig) alone falls back to
    a text repr unless the inline backend registered its PNG formatter, so
    the bytes are encoded here instead.
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


def fmt(v: float) -> str:
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3f}"


def county_mean(base: Path, stem: str, year: Optional[int]) -> pd.DataFrame:
    """One WCC per county. A rate averages over months; it does not sum."""
    hits = [p for p in base.glob(f"wcc_county_{stem}_*.feather")
            if "annual" not in p.name]
    if not hits:
        raise FileNotFoundError(
            f"no wcc_county_{stem}_*.feather in {base}. "
            f"Run wcc_ann_downscale.py first.")
    df = pd.read_feather(sorted(hits)[-1])
    if year is not None:
        df = df[df["year"] == year]
        if df.empty:
            raise ValueError(f"{stem}: no rows for {year}")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df.groupby("fips", as_index=False)["wcc_ann_l_d"].mean()


def load_counties(shapefile: Optional[str]):
    import geopandas as gpd
    g = gpd.read_file(shapefile if shapefile else COUNTY_URL)
    g = g[~g["STATEFP"].isin(NON_CONUS_FIPS)].copy()
    g["fips"] = g["GEOID"].astype(str).str.zfill(5)
    return g.to_crs("EPSG:4326")


def plot(data_dir: str = DATA_DIR, year: Optional[int] = None,
         shapefile: Optional[str] = None, cmap: str = CMAP,
         clip_pct: float = CLIP_PCT, units: str = "l"):
    """Draw the 2 x 2 map and return the figure. Nothing is saved."""
    base = Path(data_dir).expanduser()
    counties = load_counties(shapefile)
    cm = plt.get_cmap(cmap)
    k = GAL_PER_L if units == "gal" else 1.0
    unit = ("gal head$^{-1}$ d$^{-1}$" if units == "gal"
            else "L head$^{-1}$ d$^{-1}$")

    # PlateCarree with the aspect left to cartopy. Lambert Conformal plus
    # set_aspect("auto") was stretching each panel to fill its axes, which
    # is what distorted the earlier layout: on a choropleth the eye reads
    # area as magnitude, so the geometry has to stay honest.
    fig, axes = plt.subplots(2, 2, figsize=(12, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    axes = axes.ravel()

    for ax, (stem, title) in zip(axes, PANELS):
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

        vals = county_mean(base, stem, year)
        vals["wcc_ann_l_d"] *= k
        g = counties.merge(vals, on="fips", how="left")
        v = g["wcc_ann_l_d"]
        norm = make_norm(v.values, clip_pct)

        # Counties beneath the linework so boundaries stay legible on fill.
        g.plot(column="wcc_ann_l_d", cmap=cm, norm=norm, ax=ax,
               edgecolor="none", transform=ccrs.PlateCarree(), zorder=5,
               missing_kwds={"color": "lightgrey", "edgecolor": "none"})
        ax.add_feature(cfeature.STATES, edgecolor="gray",
                       linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=6)
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        # One bar per panel: a dairy cow drinks about 500 times what a
        # chicken does, so a shared scale would flatten three of the four.
        cb = fig.colorbar(sm, ax=ax, shrink=0.82, aspect=22, pad=0.02,
                          extend="both")
        cb.set_label(f"WCC [{unit}]", fontsize=10)
        cb.ax.tick_params(labelsize=8)

        n_missing = int(v.isna().sum())
        ax.set_title(f"{title} (n={int(v.notna().sum()):,})",
                     fontsize=13, fontweight="bold", loc="center")

        print(f"  {title:<14}{'climatology' if year is None else year}  "
              f"n={int(v.notna().sum()):,}  min {fmt(np.nanmin(v))}  "
              f"median {fmt(np.nanmedian(v))}  max {fmt(np.nanmax(v))}  |  "
              f"scale [{fmt(norm.vmin)} .. {fmt(norm.vmax)}]"
              + (f"   MISSING {n_missing}" if n_missing else ""))

    # fig.suptitle("County-level water consumption coefficient"
    #              + (f", {year}" if year else ", long-term mean"),
    #              fontsize=15, fontweight="bold")
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="County WCC maps, one panel per livestock type.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--shapefile", default=None)
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--clip-pct", type=float, default=CLIP_PCT)
    ap.add_argument("--units", choices=("l", "gal"), default="l")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.year, args.shapefile, args.cmap,
         args.clip_pct, args.units)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()