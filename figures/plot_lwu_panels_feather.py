"""
plot_lwu_panels_feather.py

The same four-block figure as plot_lwu_panels.py, but built from the
repository tables in data/ rather than the archived netCDF:

    a) WCC    b) WC    c) consumption-ratio    d) WW

The figure is DISPLAYED. Nothing is written to disk.

    data/wcc_county/wcc_county_<type>_*.feather      wcc_ann_l_d
    data/wc_county/county_level_WC_<type>.feather    wc_mgal_d
    data/ratio_county/county_level_ratios_<type>.feather   wc_ww_ratio

WW is not stored as a table: validate_against_usgs.py computes it in
memory and keeps only the comparison. It is therefore derived here as
WC / ratio, the same way, and county-years whose ratio falls below
MIN_RATIO are left as missing rather than producing an implausible
withdrawal from a near-zero divisor.

Rows are county-month in the WCC and WC tables, so they are averaged to
one value per county-year: both are rates, and an annual figure is the
mean daily rate, not the sum of twelve monthly rates.

Usage
-----
    from plot_lwu_panels_feather import plot, plot_all
    plot_all()
    plot("dairy_cattle", years=[2002, 2007, 2012, 2017, 2022])
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_REPO = "/scratch/hdagne1/LivestockWaterUse"
CONUS_EXTENT = [-125, -66, 24, 50]
NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
              "cb_2018_us_county_20m.zip")

LIVESTOCK = [("dairy_cattle", "Dairy cattle"), ("beef_cattle", "Beef cattle"),
             ("hogs", "Hogs"), ("poultry", "Poultry")]

# Below this the ratio is an unreliable divisor and WW is left missing.
MIN_RATIO = 0.05

BLOCKS = [("wcc", "a", "WCC [L/head/day]"),
          ("wc", "b", "WC [Mgal/day]"),
          ("ratio", "c", "Ratio [-]"),
          ("ww", "d", "WW [Mgal/day]")]

DEFAULT_YEARS = (2002, 2007, 2012, 2017, 2022)

_counties = None


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


def load_counties(shapefile: Optional[str] = None):
    """Cached, since the same geometry serves every panel and every type."""
    global _counties
    if _counties is None:
        import geopandas as gpd
        g = gpd.read_file(shapefile if shapefile else COUNTY_URL)
        g = g[~g["STATEFP"].isin(NON_CONUS_FIPS)].copy()
        g["fips"] = g["GEOID"].astype(str).str.zfill(5)
        _counties = g.to_crs("EPSG:4326")
    return _counties


def load_tables(name: str, repo: str = _REPO) -> pd.DataFrame:
    """
    One row per county-year, carrying all four quantities.

    WCC and WC are county-month rates, so they average over months; the
    ratio is already annual. WW is derived rather than read, because the
    pipeline does not store it.
    """
    base = Path(repo).expanduser() / "data"

    hits = [p for p in (base / "wcc_county").glob(f"wcc_county_{name}_*.feather")
            if "annual" not in p.name]
    if not hits:
        raise FileNotFoundError(f"no wcc_county_{name}_*.feather")
    wcc = pd.read_feather(sorted(hits)[-1])
    wcc["fips"] = wcc["fips"].astype(str).str.zfill(5)
    wcc = (wcc.groupby(["year", "fips"], as_index=False)["wcc_ann_l_d"]
           .mean().rename(columns={"wcc_ann_l_d": "wcc"}))

    p = base / "wc_county" / f"county_level_WC_{name}.feather"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    wc = pd.read_feather(p)
    wc["fips"] = wc["fips"].astype(str).str.zfill(5)
    wc = (wc.groupby(["year", "fips"], as_index=False)["wc_mgal_d"]
          .mean().rename(columns={"wc_mgal_d": "wc"}))

    p = base / "ratio_county" / f"county_level_ratios_{name}.feather"
    if not p.exists():
        p = base / "ratio_county" / "county_level_ratios_all.feather"
    ratio = pd.read_feather(p)
    ratio["fips"] = ratio["fips"].astype(str).str.zfill(5)
    ratio = (ratio[["year", "fips", "wc_ww_ratio"]]
             .drop_duplicates(["year", "fips"])
             .rename(columns={"wc_ww_ratio": "ratio"}))

    df = wcc.merge(wc, on=["year", "fips"], how="outer") \
            .merge(ratio, on=["year", "fips"], how="outer")

    # WW = WC / ratio, left missing where the divisor is unreliable.
    r = df["ratio"].where(df["ratio"] >= MIN_RATIO)
    df["ww"] = df["wc"] / r
    return df


def plot(name: str = "dairy_cattle", years: Optional[Sequence[int]] = None,
         repo: str = _REPO, shapefile: Optional[str] = None,
         cmap: str = "bwr", panel_w: float = 2.6, panel_h: float = 1.35,
         ncol: int = 3, wspace: float = 0.02, hspace: float = 0.25,
         block_gap: float = 0.045, title: Optional[str] = None):
    """
    Draw the four blocks for one livestock type and return the figure.

    panel_w and panel_h set one map panel in inches; the figure size
    follows. Because the maps use set_aspect("auto") they stretch to fill
    whatever box they are given, so changing these alters only the
    figure's width and height and no gaps open between columns.
    """
    df = load_tables(name, repo)
    years = list(years or [y for y in DEFAULT_YEARS
                           if y in set(df["year"].unique())])
    missing = [y for y in years if y not in set(df["year"].unique())]
    if missing:
        raise ValueError(f"years not in the tables: {missing}")

    counties = load_counties(shapefile)
    cm = plt.get_cmap(cmap)
    nrow = int(np.ceil(len(years) / ncol))
    title = title or dict(LIVESTOCK).get(name, name)

    fig = plt.figure(figsize=(panel_w * ncol + 0.9,
                              panel_h * nrow * len(BLOCKS)
                              + 0.35 * len(BLOCKS)), dpi=200)
    top_margin = 0.965
    block_h = ((top_margin - block_gap * (len(BLOCKS) - 1)) / len(BLOCKS))

    for bi, (col, letter, cbl) in enumerate(BLOCKS):
        top = top_margin - bi * (block_h + block_gap)
        gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.88,
                              top=top, bottom=top - block_h,
                              wspace=wspace, hspace=hspace)

        sub = df[df["year"].isin(years)][col]
        v = sub[np.isfinite(sub)]
        vmin = float(v.min()) if len(v) else 0.0
        vmax = float(v.max()) if len(v) else 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        axes = []
        for k, year in enumerate(years):
            ax = fig.add_subplot(gs[k // ncol, k % ncol],
                                 projection=ccrs.LambertConformal())
            ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
            ax.set_aspect("auto")

            ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
            ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
            ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                           edgecolor="#f0f2f4", linewidth=0.3)

            year_df = df[df["year"] == year][["fips", col]]
            g = counties.merge(year_df, on="fips", how="left")
            g.plot(column=col, cmap=cm, norm=norm, ax=ax, edgecolor="none",
                   transform=ccrs.PlateCarree(), zorder=5,
                   missing_kwds={"color": "white", "edgecolor": "none"})

            ax.add_feature(cfeature.STATES, edgecolor="#888888",
                           linewidth=0.35, zorder=6)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
                           linewidth=0.5, zorder=6)
            ax.set_title(f"{year}", fontsize=9)
            axes.append(ax)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, shrink=0.95, aspect=26, pad=0.015,
                          fraction=0.03)
        cb.set_label(cbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)

        axes[0].text(-0.10, 1.18, f"{letter})", transform=axes[0].transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

        print(f"  {letter}) {col:<8} range [{vmin:.4g}, {vmax:.4g}]  "
              f"mean {v.mean():.4g}")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    _show(fig)
    plt.close(fig)
    return fig


def plot_all(repo: str = _REPO, **kw):
    """One figure per livestock type, drawn in sequence."""
    figs = {}
    for name, label in LIVESTOCK:
        print(f"\n{label}")
        figs[label] = plot(name, repo=repo, title=label, **kw)
    return figs