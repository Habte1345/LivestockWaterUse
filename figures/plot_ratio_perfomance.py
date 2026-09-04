"""
plot_ratio_performance.py

Performance of the second ML: downscaling the state-level WC-to-WW
consumption ratios to county level.

Two panels:

    a) cross-validated on the state-level training data, whole states held
       out, with the training curve inset
    b) predicted county ratios against the USGS county-level ratios for
       1985, 1990 and 1995, which the model never saw

The figure is DISPLAYED. Nothing is written to disk.

The two panels answer different questions and are shown side by side for
that reason. Panel (a) asks whether the relationship transfers to states
absent from fitting; panel (b) asks whether it transfers to a finer
spatial scale and a later period at once, which is the harder test and the
one that matters for the product.

Reads from ratio_ann_downscale.py and validate_against_usgs.py.

Usage
-----
    from plot_ratio_performance import plot
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

_REPO = "/scratch/hdagne1/LivestockWaterUse"
RATIO_DIR = f"{_REPO}/data/ratio_county"
VALID_DIR = f"{_REPO}/data/validation"


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


def density(x, y, bins: int = 80):
    """
    Local point density on a log scale.

    Two thirds of the observed ratios sit at exactly 1.0, so one cell holds
    thousands of points while the rest hold a handful. On a linear colour
    scale that single cell saturates the map and everything else renders
    identically, leaving the eye to read only the sparse outliers.
    """
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    ix = np.clip(np.digitize(x, xe) - 1, 0, h.shape[0] - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, h.shape[1] - 1)
    return np.log10(h[ix, iy] + 1.0)


def panel(ax, obs, pred, title, xlab, ylab, cmap, size):
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]

    c = density(obs, pred)
    o = np.argsort(c)
    ax.scatter(obs[o], pred[o], c=c[o], cmap=cmap, s=size, linewidths=0,
               rasterized=True)

    # The USGS county file contains a small number of ratios far above 1,
    # reaching 8.0 in the 1985 Florence and Forest County, Wisconsin
    # records. A ratio above 1 means consuming more water than was
    # withdrawn, so these are source errors. Scaling the axes to them
    # compresses every real point into the corner and makes the panel
    # unreadable, so the view is capped just past the physical bound and
    # the number left off scale is stated.
    lo = float(max(0.0, np.nanpercentile(np.concatenate([obs, pred]), 0.5)
                   - 0.05))
    hi = 1.1
    pad = 0.0
    off = int(((obs > hi) | (pred > hi) | (obs < lo) | (pred < lo)).sum())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
    # The ratio is bounded above by unity: consumption cannot exceed
    # withdrawal, so the region beyond this line is physically impossible.
    ax.axhline(1.0, color="0.45", lw=0.9, ls=":")
    ax.axvline(1.0, color="0.45", lw=0.9, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if off:
        ax.text(0.97, 0.03, f"{off:,} point(s) off scale",
                transform=ax.transAxes, fontsize=8, ha="right",
                va="bottom", color="0.35")

    # Statistics use every point, including those off scale: the view is
    # clipped, the evaluation is not.
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1 - float(np.sum((obs - pred) ** 2)) / ss_tot if ss_tot else np.nan
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    bias = float(np.mean(pred - obs))
    err = np.abs(pred - obs)

    ax.text(0.04, 0.96,
            f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.3f}\nbias = {bias:+.3f}\n"
            f"n = {len(obs):,}",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    return {"r2": r2, "rmse": rmse, "bias": bias, "n": len(obs),
            "median_abs_err": float(np.median(err))}


def plot(ratio_dir: str = RATIO_DIR, valid_dir: str = VALID_DIR,
         cmap: str = "jet", size: float = 5.0):
    """Draw the two panels and return the figure. Nothing is saved."""
    rd, vd = Path(ratio_dir).expanduser(), Path(valid_dir).expanduser()

    oof = rd / "ratio_ann_oof.feather"
    hist = rd / "ratio_ann_training_history.feather"
    test = vd / "validation_ratio_county.feather"
    if not test.exists():
        raise FileNotFoundError(
            f"{test} not found. Run validate_against_usgs.py first.")

    t = pd.read_feather(test)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # Panel a: cross-validated on the state-level training data.
    if oof.exists():
        d = pd.read_feather(oof)
        sa = panel(axes[0], d["observed"], d["ann"],
                   "a) Cross-validated, state level, 1960-1980",
                   "Observed state ratio [-]", "Predicted ratio [-]",
                   cmap, size * 3)
        if hist.exists():
            h = pd.read_feather(hist)
            if not h.empty:
                ins = axes[0].inset_axes([0.58, 0.10, 0.38, 0.30])
                ins.plot(h["epoch"], h["train_rmse"], color="tab:blue",
                         lw=1.3, label="Train")
                if "val_rmse" in h:
                    ins.plot(h["epoch"], h["val_rmse"], color="tab:orange",
                             lw=1.3, ls="--", label="Valid")
                    ins.axvline(h.loc[h["val_rmse"].idxmin(), "epoch"],
                                color="0.4", lw=0.8, ls=":")
                ins.set_xlabel("Epoch", fontsize=8, labelpad=1)
                ins.set_ylabel("RMSE", fontsize=8, labelpad=1)
                ins.tick_params(labelsize=7)
                ins.legend(fontsize=7, frameon=True)
                ins.grid(alpha=0.25, lw=0.4)
    else:
        sa = None
        axes[0].text(0.5, 0.5, "no cross-validation output found",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("a) Cross-validated, state level",
                          fontsize=13, fontweight="bold")

    # Panel b: against the USGS county ratios, unseen in training.
    sb = panel(axes[1], t["observed"], t["modelled"],
               "b) Test, county level, 1985-1995",
               "USGS observed county ratio [-]", "Predicted ratio [-]",
               cmap, size)

    if sa:
        print(f"  a) state CV    R2 {sa['r2']:+.4f}  RMSE {sa['rmse']:.4f}  "
              f"bias {sa['bias']:+.4f}  n {sa['n']:,}")
    print(f"  b) county test R2 {sb['r2']:+.4f}  RMSE {sb['rmse']:.4f}  "
          f"bias {sb['bias']:+.4f}  n {sb['n']:,}")
    print(f"     median |error| {sb['median_abs_err']:.4f}")

    plt.tight_layout()
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Performance of the consumption-ratio downscaling.",
        allow_abbrev=False)
    ap.add_argument("--ratio-dir", default=RATIO_DIR)
    ap.add_argument("--valid-dir", default=VALID_DIR)
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--size", type=float, default=5.0)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.ratio_dir, args.valid_dir, args.cmap, args.size)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()