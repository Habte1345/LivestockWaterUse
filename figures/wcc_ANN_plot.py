"""
plot_wcc_ann.py

ANN-transferred WCC against the MLR surrogate target, one panel per
livestock type, with the training and validation curves as an inset.

The figure is DISPLAYED. Nothing is written to disk.

Reads two tables written by wcc_ann_downscale.py:

    wcc_ann_predictions.feather        per-sample target and prediction,
                                       labelled train / validation / test
    wcc_ann_training_history.feather   per-epoch train and validation RMSE

Points are coloured by local density, so the bulk of the cloud is visible
rather than being hidden under overplotting -- with tens of thousands of
samples a plain scatter is a solid blob.

Usage
-----
    python plot_wcc_ann.py
    python plot_wcc_ann.py --split validation --units gal
    from plot_wcc_ann import plot; plot()
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
DATA_DIR = f"{_REPO}/data/wcc_county"

PANELS = [
    ("dairy_cattle", "Dairy cattle"),
    ("beef_cattle", "Beef cattle"),
    ("hogs", "Hogs"),
    ("poultry", "Poultry"),
]

GAL_PER_L = 1.0 / 3.785411784


def _show(fig, screen_dpi: int = 130) -> None:
    """
    Render whatever the kernel is configured to do.

    plt.show() is a silent no-op when the backend is non-interactive, and
    display(fig) alone is no better, because matplotlib's PNG formatter is
    registered only by the inline backend. Publishing the encoded bytes
    depends on neither.
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


def density_colour(x, y, bins: int = 220):
    """
    Local point density from a 2-D histogram lookup.

    Far cheaper than a gaussian KDE at this sample size, and the visual
    result is the same: the dense core reads warm, the tails cool.
    """
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    ix = np.clip(np.digitize(x, xe) - 1, 0, h.shape[0] - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, h.shape[1] - 1)
    return h[ix, iy]


def fmt(v: float) -> str:
    """Poultry WCC is around 0.15 L/d, so fixed precision would print 0.00."""
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def plot(data_dir: str = DATA_DIR, split: str = "both", units: str = "l",
         cmap: str = "bwr", point_size: float = 3.0):
    """
    Draw the 2 x 2 panel and return the figure. Nothing is saved.

    split="both" pools train and validation, which is what the reference
    figure shows; "validation" or "test" restricts to one, and the test
    split is the honest one to quote, since neither the fitting nor the
    stopping rule ever saw it.
    """
    base = Path(data_dir).expanduser()
    pred_path = base / "wcc_ann_predictions.feather"
    hist_path = base / "wcc_ann_training_history.feather"
    for p in (pred_path, hist_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run wcc_ann_downscale.py first.")

    preds = pd.read_feather(pred_path)
    hist = pd.read_feather(hist_path)

    k = GAL_PER_L if units == "gal" else 1.0
    unit_label = ("gal head$^{-1}$ day$^{-1}$" if units == "gal"
                  else "L head$^{-1}$ day$^{-1}$")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    axes = axes.flatten()

    for ax, (stem, title) in zip(axes, PANELS):
        d = preds[preds["livestock_type"] == stem]
        if split != "both":
            d = d[d["split"] == split]
        else:
            d = d[d["split"].isin(["train", "validation"])]
        if d.empty:
            ax.text(0.5, 0.5, f"no rows for {stem}", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight="bold")
            continue

        x = d["wcc_surrogate_l_d"].to_numpy(dtype=float) * k
        y = d["wcc_ann_l_d"].to_numpy(dtype=float) * k
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        c = density_colour(x, y)
        order = np.argsort(c)          # densest points drawn on top
        ax.scatter(x[order], y[order], c=c[order], cmap=cmap,
                   s=point_size, linewidths=0, rasterized=True)

        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.02 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((x - x.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))

        ax.text(0.04, 0.95, f"$R^2$={r2:.2f}\nRMSE={fmt(rmse)}",
                transform=ax.transAxes, fontsize=13,
                verticalalignment="top")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Surrogate WCC [{unit_label}]", fontsize=11)
        ax.set_ylabel(f"ANN-transferred WCC [{unit_label}]", fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)

        # Inset: the learning curves for this type.
        h = hist[hist["livestock_type"] == stem]
        if not h.empty:
            ins = ax.inset_axes([0.56, 0.09, 0.40, 0.30])
            ins.plot(h["epoch"], h["train_rmse_l_d"] * k,
                     color="tab:blue", lw=1.4, label="Train")
            ins.plot(h["epoch"], h["val_rmse_l_d"] * k,
                     color="tab:orange", lw=1.4, ls="--", label="Valid")
            best = h.loc[h["val_rmse_l_d"].idxmin(), "epoch"]
            ins.axvline(best, color="0.4", lw=0.8, ls=":")
            ins.set_xlabel("Epoch", fontsize=8, labelpad=1)
            ins.set_ylabel("RMSE", fontsize=8, labelpad=1)
            ins.tick_params(labelsize=7)
            ins.legend(fontsize=7, frameon=True, loc="upper right")
            ins.grid(alpha=0.25, lw=0.4)

        print(f"  {title:<14} {split:<11} n={len(x):,}  "
              f"R2={r2:.4f}  RMSE={fmt(rmse)}  "
              f"bias={fmt(float(np.mean(y - x)))}"
              + (f"  best epoch {int(best)}" if not h.empty else ""))

    plt.tight_layout()
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="ANN-transferred WCC against the MLR surrogate target.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--split", choices=("both", "train", "validation", "test"),
                    default="both",
                    help="both pools train and validation; test is the "
                         "split neither fitting nor early stopping saw")
    ap.add_argument("--units", choices=("l", "gal"), default="l")
    ap.add_argument("--cmap", default="bwr")
    ap.add_argument("--point-size", type=float, default=3.0)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.split, args.units, args.cmap, args.point_size)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()