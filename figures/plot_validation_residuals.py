"""
plot_validation_residuals.py

Spatial residuals of the pathway against the USGS observations, one panel
per quantity and testing year, on a cartopy Lambert Conformal basemap with
discrete binned colourbars.

    withdrawal WW   [Mgal/d]   2005, 2010, 2015
    ratio WC:WW     [-]        1985, 1990, 1995
    consumption WC  [Mgal/d]   no overlapping years, so not drawn

Residual is MODELLED minus OBSERVED, so on RdYlGn red is under-prediction
and green over-prediction.

The figure is DISPLAYED. Nothing is written to disk.

Each quantity gets its own colour scale. WW is in Mgal/d and spans orders
of magnitude; the ratio is dimensionless and near zero. A shared scale
would flatten the ratio panels completely.

The scale is symmetric about zero and trimmed at the 2nd and 98th
percentile of |residual|, so the colour reads as sign and a few extreme
counties cannot absorb the whole ramp.

Usage
-----
    python plot_validation_residuals.py
    python plot_validation_residuals.py --bins 8
    from plot_validation_residuals import plot; plot()
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
DATA_DIR = f"{_REPO}/data/validation"

NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_EXTENT = [-125, -66, 24, 50]
COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
              "cb_2018_us_county_20m.zip")

# (file stem, short label, colourbar label)
QUANTITIES = [
    ("withd", "WW", "WW residual [Mgal d$^{-1}$]"),
    ("ratio", "Ratio", "Ratio residual [-]"),
    ("consu", "WC", "WC residual [Mgal d$^{-1}$]"),
]

CMAP, N_BINS, CLIP_PCT = "bwr", 6, 2.0


def make_boundary_norm(values, cmap, n_bins=N_BINS, clip_pct=CLIP_PCT):
    """
    Discrete classes, symmetric about zero.

    Symmetric because a residual has a sign: the colour should say which
    way the model is wrong, not merely how much. Trimmed at the 2nd and
    98th percentile of the absolute residual so a handful of extreme
    counties cannot take the entire ramp.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return mcolors.BoundaryNorm(np.linspace(-1, 1, n_bins + 1), cmap.N)
    lim = float(np.nanpercentile(np.abs(v), 100 - clip_pct))
    if lim <= 0:
        lim = float(np.nanmax(np.abs(v))) or 1.0
    return mcolors.BoundaryNorm(np.linspace(-lim, lim, n_bins + 1), cmap.N)


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


def load_counties(shapefile: Optional[str]):
    import geopandas as gpd
    g = gpd.read_file(shapefile if shapefile else COUNTY_URL)
    g = g[~g["STATEFP"].isin(NON_CONUS_FIPS)].copy()
    g["fips"] = g["GEOID"].astype(str).str.zfill(5)
    return g.to_crs("EPSG:4326")


def load_panels(base: Path):
    """Every (quantity, year) pair the validation step actually produced."""
    panels, missing = [], []
    for stem, short, cbl in QUANTITIES:
        p = base / f"validation_{stem}_county.feather"
        if not p.exists():
            missing.append(short)
            continue
        d = pd.read_feather(p)
        if d.empty:
            missing.append(short)
            continue
        d["fips"] = d["fips"].astype(str).str.zfill(5)
        d["residual"] = d["modelled"] - d["observed"]
        # One scale per quantity, shared across its years so the panels of
        # a row are comparable with each other.
        norm = make_boundary_norm(d["residual"].values, plt.get_cmap(CMAP))
        for y in sorted(int(v) for v in d["year"].unique()):
            panels.append((short, cbl, y, d[d["year"] == y], norm))
    return panels, missing


def plot(data_dir: str = DATA_DIR, shapefile: Optional[str] = None,
         cmap: str = CMAP, bins: int = N_BINS, clip_pct: float = CLIP_PCT,
         ncol: int = 3, panel_h: float = 2.6):
    """Draw the residual maps and return the figure. Nothing is saved."""
    base = Path(data_dir).expanduser()
    panels, missing = load_panels(base)
    if not panels:
        raise FileNotFoundError(
            f"no validation_*_county.feather in {base}. "
            f"Run validate_against_usgs.py first.")

    counties = load_counties(shapefile)
    cm = plt.get_cmap(cmap)
    nrow = int(np.ceil(len(panels) / ncol))

    # Sized from the panel count so the maps are never squeezed by the
    # colourbars: each panel gets a fixed box and tight_layout only has to
    # place them, not resolve a conflict.
    #
    # set_aspect("auto") stretches each map to fill its box, so height and
    # width have to move together. Changing panel_h alone squashes the
    # maps; the default keeps the 1.61 width-to-height ratio the layout was
    # built around, and --panel-h rescales panel_w to preserve it.
    fig = plt.figure(figsize=(panel_h * 2.61 * ncol, panel_h * nrow),
                     dpi=200)

    for i, (short, cbl, year, d, norm) in enumerate(panels, start=1):
        ax = fig.add_subplot(nrow, ncol, i, projection=ccrs.LambertConformal())
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        ax.set_aspect("auto")

        ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
        ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0",
                       edgecolor="#8fa8b8", linewidth=0.4)

        g = counties.merge(d[["fips", "residual"]], on="fips", how="left")
        g.plot(column="residual", cmap=cm, norm=norm, ax=ax,
               edgecolor="none", transform=ccrs.PlateCarree(), zorder=5,
               missing_kwds={"color": "white", "edgecolor": "none"})

        ax.add_feature(cfeature.STATES, edgecolor="#888888",
                       linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555",
                       linewidth=0.7, zorder=6)
        ax.add_feature(cfeature.RIVERS, edgecolor="#7fb3d5",
                       linewidth=0.4, alpha=0.5, zorder=6)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.96, pad=0.02, extend="both")
        cbar.set_label(cbl, fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        v = d["residual"].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.set_title(f"{short} {year}  (n={len(v):,})",
                     fontsize=12, fontweight="bold")

        # Residual distribution inset. The map shows where the model is
        # wrong; the histogram shows by how much and in which direction.
        # Both are needed: a map of mostly pale cells could be a tight
        # distribution centred on zero or a broad one that happens to
        # cancel, and only the histogram separates those.
        if v.size:
            # Lower right, rotated: the histogram's value axis runs
            # vertically, parallel with the colourbar, so the two scales
            # read together instead of competing. orientation="horizontal"
            # on hist() puts the residual on the y axis.
            ins = ax.inset_axes([0.78, 0.04, 0.19, 0.42])
            lim = float(norm.boundaries[-1])
            ins.hist(np.clip(v, -lim, lim), bins=20, color="blue", alpha=0.6,
                     edgecolor="black", linewidth=0.4,
                     orientation="horizontal")
            mean_val = float(np.mean(v))
            ins.axhline(np.clip(mean_val, -lim, lim), color="magenta",
                        linestyle="--", linewidth=1.5)
            ins.axhline(0.0, color="0.35", linewidth=0.8)
            ins.set_ylim(-lim, lim)
            # Bars grow away from the colourbar, so the count axis points
            # left and the residual axis sits on the colourbar side. The
            # count limit is stretched to 2.2x the tallest bar, which
            # leaves the left of the box empty for the label -- putting the
            # label anywhere inside the bar region overlaps it whatever the
            # coordinate.
            top = max(ins.get_xlim())
            ins.set_xlim(top * 2.2, 0)
            # Above the inset box, in axes coordinates: anywhere inside
            # the box can collide with a bar, because bar lengths depend on
            # the data and cannot be avoided by choosing a fixed position.
            ins.text(0.5, 0.5, f"{mean_val:.3g}", color="magenta",
                     ha="center", va="bottom", fontsize=10,
                     transform=ins.transAxes)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.patch.set_alpha(0.85)

        # print(f"  {short:<6}{year}  n={len(v):>6,}  "
        #       f"mean {np.mean(v):+.4g}  median {np.median(v):+.4g}  "
        #       f"RMSE {np.sqrt(np.mean(v**2)):.4g}")

    # if missing:
    #     print(f"  not drawn: {', '.join(missing)} "
    #           f"-- no overlapping years with the USGS record")

    plt.tight_layout()
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spatial residual maps against USGS.", allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--shapefile", default=None)
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--clip-pct", type=float, default=CLIP_PCT)
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--panel-h", type=float, default=2.6,
                    help="height of one map panel in inches; width follows "
                         "at 1.61x so the maps do not distort")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.shapefile, args.cmap, args.bins,
         args.clip_pct, args.ncol, args.panel_h)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()