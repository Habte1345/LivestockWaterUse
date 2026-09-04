"""
plot_wcc_regression.py

Literature-based WCC against the fitted MLR surrogate, one panel per
livestock type, in the same style as the reference figure: hexbin density
with the jet colormap, a red fitted line, a dashed black 1:1 line, and an
R-squared / RMSE annotation.

Unlike the mock-up this replaces, nothing here is simulated. Both axes come
from wcc_surrogate_<type>.feather written by wcc_mlr.py:

    x = wcc_literature_gal_d   the published intake equation for that sample
    y = wcc_mlr_gal_d          the fitted surrogate applied to the same sample

Those rows are an independent draw, out of sample with respect to the fit,
so the agreement shown is genuine out-of-sample agreement rather than a
model being compared against itself.

Two statistics are annotated:
    R2, RMSE   for the least-squares line of y on x, matching the reference
               figure exactly
    R2(1:1)    agreement about the identity line, which is the quantity that
               actually matters when asking whether the surrogate reproduces
               the literature equation. A model can sit on a tight but
               displaced line and score well on the first while failing the
               second.

The figure is DISPLAYED, not written to disk. Pass --save <path> (or
save=<path> to plot()) if a file is wanted as well.

Usage
-----
    python plot_wcc_regression.py
    python plot_wcc_regression.py --save figures/Regression.png

    from plot_wcc_regression import plot
    plot()
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# The backend is deliberately left alone. Forcing Agg here would stop the
# figure rendering inline under %run in a notebook, which is the normal way
# this script is used. Headless callers can still pass --save.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

_REPO = "/scratch/hdagne1/LivestockWaterUse"
DATA_DIR = f"{_REPO}/data/wcc_mlr"

# (file stem, panel title) in the order the panels are drawn
PANELS = [
    ("dairy_cattle", "Dairy cattle"),
    ("beef_cattle", "Beef cattle"),
    ("hogs", "Hogs"),
    ("poultry", "Poultry"),
]

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams["axes.edgecolor"] = "0.3"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["grid.color"] = "0.88"
plt.rcParams["grid.linewidth"] = 0.6

# bwr is a diverging map, so on a linear count scale nearly every cell
# sits in its white middle and the panel washes out. The density is put on
# a log scale so the colour spans the orders of magnitude actually present
# between the sparse tails and the dense core.
CMAP = "bwr"


def _show(fig, screen_dpi: int = 110) -> None:
    """
    Render the figure, whatever the kernel is configured to do.

    Two ways this silently fails otherwise:

      * plt.show() does nothing at all when the active backend is
        non-interactive, which is the default in a kernel where
        `%matplotlib inline` was never run. No figure, no error.
      * IPython.display.display(fig) is no better on its own: matplotlib's
        PNG formatter is only registered by the inline backend, so without
        it IPython falls back to the text representation and prints
        "<Figure size 1000x1000 with 4 Axes>".

    Encoding the PNG here and publishing the bytes depends on neither. If
    there is no IPython at all, fall back to plt.show() for a desktop
    window.
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
    # Closed so the inline backend, if it IS active, does not draw a second
    # copy when the cell ends.
    plt.close(fig)


def fmt(v: float) -> str:
    """
    Poultry WCC is around 0.04 gal/d, so a fixed two-decimal RMSE would
    print 0.00 and say nothing. Scale the precision to the magnitude.
    """
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def plot(data_dir: str = DATA_DIR, save: Optional[str] = None,
         dpi: int = 600, gridsize: int = 90, units: str = "gal",
         expected: bool = False, cmap: str = CMAP):
    """
    Draw the figure and return it. Nothing is written to disk unless `save`
    is given a path.

        from plot_wcc_regression import plot
        plot()                                   # display only
        plot(save="figures/Regression.png")      # display and save

    Calling this bypasses argparse, which is the safer route in Jupyter.
    """
    if save is not None and not isinstance(save, (str, Path)):
        raise TypeError(
            f"save must be a path or None, got {type(save).__name__} "
            f"({save!r}). This usually means plot() was called with "
            f"positional arguments in the wrong order -- the signature is "
            f"plot(data_dir, save, dpi, gridsize, units, expected). Use "
            f"keyword arguments: plot(data_dir=..., gridsize=...).")

    base = Path(data_dir).expanduser()

    suffix = "gal_d" if units == "gal" else "l_d"
    unit_label = ("gal day$^{-1}$ head$^{-1}$" if units == "gal"
                  else "L day$^{-1}$ head$^{-1}$")

    # constrained_layout, not subplots_adjust: the panels are forced to an
    # equal aspect so that the 1:1 line is at 45 degrees, which makes their
    # drawn size depend on the data range. Fixed spacing then leaves the
    # x-labels of the top row sitting on the panels beneath.
    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                             constrained_layout=True)
    axes = axes.flatten()

    missing = []
    for idx, (ax, (stem, title)) in enumerate(zip(axes, PANELS)):
        # Both quantities are shared across the 2 x 2 grid, so the y label
        # goes on the left column only and the x label on the bottom row
        # only.
        show_y = idx % 2 == 0
        show_x = idx >= 2

        path = base / f"wcc_surrogate_{stem}.feather"
        if not path.exists():
            missing.append(path.name)
            ax.text(0.5, 0.5, f"{path.name}\nnot found",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11)
            ax.set_title(title, fontsize=14, fontweight="bold")
            continue

        df = pd.read_feather(path)
        # The x axis is the realised literature-based WCC for each sampled
        # county-year herd, which is what the surrogate is fitted to. The
        # noise-free equation value is also in the file as
        # wcc_expected_*, and --against expected plots that instead.
        xcol = (f"wcc_expected_{suffix}" if expected
                else f"wcc_literature_{suffix}")
        if xcol not in df:
            xcol = f"wcc_literature_{suffix}"
        literature = df[xcol].to_numpy(dtype=float)
        regression = df[f"wcc_mlr_{suffix}"].to_numpy(dtype=float)

        ok = np.isfinite(literature) & np.isfinite(regression)
        literature, regression = literature[ok], regression[ok]

        X = literature.reshape(-1, 1)
        model = LinearRegression().fit(X, regression)
        y_pred = model.predict(X)

        r2 = r2_score(regression, y_pred)
        rmse = np.sqrt(mean_squared_error(regression, y_pred))
        # Agreement about the identity line, not about the fitted line.
        r2_identity = 1.0 - (np.sum((regression - literature) ** 2)
                             / np.sum((literature - literature.mean()) ** 2))

        hb = ax.hexbin(
            literature,
            regression,
            gridsize=gridsize,
            edgecolors="none",
            cmap=cmap,
            bins="log",
            mincnt=1,
        )

        x_line = np.linspace(literature.min(), literature.max(), 200)
        y_line = model.predict(x_line.reshape(-1, 1))

        ax.plot(x_line, y_line, color="red", lw=2)
        ax.plot(x_line, x_line, "--", color="black", lw=1)

        ax.set_title(title, fontsize=16, fontweight="bold")
        if show_x:
            ax.set_xlabel(f"Literature-based WCCs [{unit_label}]",
                          fontsize=16, labelpad=16)
        if show_y:
            ax.set_ylabel(f"MLR WCCs [{unit_label}]", fontsize=16,
                          labelpad=10)

        stats = (f"$R^2$ = {r2:.2f}\n"
                 f"RMSE = {fmt(rmse)}\n"
                #  f"$R^2_{{1:1}}$ = {r2_identity:.2f}\n"
                #  f"n = {len(literature):,}"
                 )
        ax.text(0.05, 0.92, stats, transform=ax.transAxes,
                fontsize=16, verticalalignment="top")

        print(f"  {title:<14} n={len(literature):>7,}  "
              f"slope={model.coef_[0]:.4f}  intercept={fmt(model.intercept_)}  "
              f"R2={r2:.4f}  RMSE={fmt(rmse)}  R2(1:1)={r2_identity:.4f}")

    # cb = fig.colorbar(hb, ax=axes.tolist(), shrink=0.99, aspect=32,
    #                   pad=0.02)
    # cb.set_label("Sample density [count per bin]", fontsize=11)
    # cb.ax.tick_params(labelsize=9)

    if missing:
        print(f"  WARNING: missing input files: {missing}")
        print("  Run wcc_mlr.py first.")

    if save:
        out = Path(save).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved {out}")

    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    # allow_abbrev=False is essential. A Jupyter kernel injects
    # --f=<connection file> into sys.argv, and with abbreviation enabled
    # argparse can match --f to an option starting with "f" and silently
    # use the kernel JSON path as its value. With abbreviation off it lands
    # in `unknown` and is ignored.
    ap = argparse.ArgumentParser(
        description="Plot MLR surrogate against literature-based WCC.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--save", default=None,
                    help="optional path to write the figure to; by default "
                         "it is only displayed")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--gridsize", type=int, default=90)
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--units", choices=("gal", "l"), default="gal")
    ap.add_argument("--against", choices=("sample", "expected"),
                    default="sample",
                    help="sample = the realised literature WCC per herd "
                         "(default); expected = the noise-free equation value")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.save, args.dpi, args.gridsize, args.units,
         args.against == "expected", args.cmap)
    return 0


if __name__ == "__main__":
    # Deliberately NOT `raise SystemExit(main())`. Under `%run` in a
    # notebook, SystemExit aborts the cell before IPython's inline backend
    # flushes the figure, so the plot is computed and then never drawn.
    main()