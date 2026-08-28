"""
plot_ratio_ann.py

Performance of the WC-to-WW ratio downscaling, as three panels:

    a) cross-validated, state level, 1960-1980, with the training curve inset
    b) independent test, county level, 1985 / 1990 / 1995
    c) the same test points for the linear baseline and for predicting 1.0,
       so the ANN is judged against something rather than in isolation

The figure is DISPLAYED. Nothing is written to disk.

Every point in (a) is an out-of-fold prediction: the state it belongs to
was never in the training set for that fold. Panel (b) is a harder test
still, transferring across both spatial scale and a twenty-year gap.

A dotted line marks 1.0, the physical upper bound of the ratio -- you
cannot consume more water than you withdraw. Points above it are
impossible and are drawn, not hidden.

Usage
-----
    python plot_ratio_ann.py
    from plot_ratio_ann import plot; plot()
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
DATA_DIR = f"{_REPO}/data/ratio_county"


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


def density(x, y, bins: int = 90):
    """Local point density from a 2-D histogram; cheap and adequate here."""
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    ix = np.clip(np.digitize(x, xe) - 1, 0, h.shape[0] - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, h.shape[1] - 1)
    return h[ix, iy]


def view_limits(obs, pred, hi_cap: float = 1.25):
    """
    Axis range for a bounded ratio.

    The USGS county file contains a small number of ratios far above 1 --
    the 1985 Florence and Forest County Wisconsin records reach 8.0, which
    is a source data error, not a real consumptive fraction. Scaling the
    axes to those values compresses every genuine point into the corner
    and makes the panel unreadable. The view is therefore capped just
    above the physical bound of 1, and the number of points left off
    scale is annotated rather than quietly dropped.
    """
    v = np.concatenate([obs, pred])
    v = v[np.isfinite(v)]
    lo = float(np.nanpercentile(v, 0.5))
    lo = min(lo, 0.0) if lo < 0 else max(0.0, lo - 0.05)
    return lo, hi_cap


def scatter_panel(ax, obs, pred, title, cmap, size, label_pred):
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = np.asarray(obs)[ok], np.asarray(pred)[ok]
    c = density(obs, pred)
    order = np.argsort(c)
    ax.scatter(obs[order], pred[order], c=c[order], cmap=cmap, s=size,
               linewidths=0, rasterized=True)

    lo, hi = view_limits(obs, pred)
    off = int(((obs > hi) | (pred > hi) | (obs < lo) | (pred < lo)).sum())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
    # The ratio cannot physically exceed 1.
    ax.axhline(1.0, color="0.45", lw=0.9, ls=":")
    ax.axvline(1.0, color="0.45", lw=0.9, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if off:
        ax.text(0.97, 0.03, f"{off:,} point(s) off scale",
                transform=ax.transAxes, fontsize=8, ha="right",
                va="bottom", color="0.35")

    # Statistics on ALL points, including the off-scale ones: the view is
    # clipped, the evaluation is not.
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    bias = float(np.mean(pred - obs))
    above = int((pred > 1.0).sum())

    ax.text(0.04, 0.96,
            f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}\nbias={bias:+.3f}\nn={len(obs):,}",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("USGS observed ratio [-]", fontsize=10)
    ax.set_ylabel(label_pred, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    return dict(r2=r2, rmse=rmse, bias=bias, n=len(obs), above_one=above)


def plot(data_dir: str = DATA_DIR, cmap: str = "jet", size: float = 6.0):
    """Draw the three panels and return the figure. Nothing is saved."""
    base = Path(data_dir).expanduser()

    # Only the test predictions are strictly required. The out-of-fold
    # table is a newer output of ratio_ann_downscale.py, so panel (a)
    # falls back to the per-fold RMSE comparison when it is absent rather
    # than refusing to draw anything.
    def optional(name):
        p = base / name
        return pd.read_feather(p) if p.exists() else pd.DataFrame()

    test = optional("ratio_ann_predictions.feather")
    if test.empty:
        raise FileNotFoundError(
            f"{base / 'ratio_ann_predictions.feather'} not found. "
            f"Run ratio_ann_downscale.py first.")
    hist = optional("ratio_ann_training_history.feather")
    oof = optional("ratio_ann_oof.feather")
    folds = optional("ratio_ann_folds.feather")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    if not oof.empty:
        s1 = scatter_panel(axes[0], oof["observed"], oof["ann"],
                           "a) Cross-validated, state level, 1960-1980",
                           cmap, size, "ANN-predicted ratio [-]")
    elif not folds.empty:
        # No per-sample predictions available: show the fold-by-fold RMSE
        # against both baselines, which carries the same message.
        x = np.arange(len(folds))
        w = 0.27
        axes[0].bar(x - w, folds["val_rmse_ann"], w, label="ANN",
                    color="tab:blue")
        axes[0].bar(x, folds["val_rmse_linear"], w, label="linear",
                    color="tab:green")
        if "val_rmse_unity" in folds:
            axes[0].bar(x + w, folds["val_rmse_unity"], w,
                        label="predict 1.0", color="tab:red")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{int(f)}" for f in folds["fold"]])
        axes[0].set_xlabel("Cross-validation fold (held-out states)",
                           fontsize=10)
        axes[0].set_ylabel("Validation RMSE [-]", fontsize=10)
        axes[0].set_title("a) Cross-validated by state, 1960-1980",
                          fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.25, lw=0.5, axis="y")
        s1 = {"r2": np.nan,
              "rmse": float(folds["val_rmse_ann"].mean()),
              "bias": np.nan, "n": int(folds["n_val"].sum())}
    else:
        axes[0].text(0.5, 0.5, "no cross-validation output found",
                     ha="center", va="center", transform=axes[0].transAxes)
        s1 = {"r2": np.nan, "rmse": np.nan, "bias": np.nan, "n": 0}

    # Training curve inset: the held-out-states fit behind the final model.
    if not hist.empty:
        ins = axes[0].inset_axes([0.60, 0.62, 0.37, 0.33])
        ins.plot(hist["epoch"], hist["train_rmse"], color="tab:blue",
                 lw=1.3, label="Train")
        if "val_rmse" in hist:
            ins.plot(hist["epoch"], hist["val_rmse"], color="tab:orange",
                     lw=1.3, ls="--", label="Valid")
            ins.axvline(hist.loc[hist["val_rmse"].idxmin(), "epoch"],
                        color="0.4", lw=0.8, ls=":")
        ins.set_xlabel("Epoch", fontsize=8, labelpad=1)
        ins.set_ylabel("RMSE", fontsize=8, labelpad=1)
        ins.tick_params(labelsize=7)
        ins.legend(fontsize=7, frameon=True)
        ins.grid(alpha=0.25, lw=0.4)

    s2 = scatter_panel(axes[1], test["ls_cw_ratio"], test["wc_ww_ratio"],
                       "b) Test, county level, 1985-1995",
                       cmap, size, "ANN-predicted ratio [-]")

    # Panel c: the same observations against the two reference predictions.
    obs = test["ls_cw_ratio"].to_numpy(float)
    ax = axes[2]
    ax.scatter(obs, np.full_like(obs, 1.0), s=size, c="tab:red", alpha=0.35,
               linewidths=0, rasterized=True, label="predict 1.0")
    ax.scatter(obs, test["wc_ww_ratio"], s=size, c="tab:blue", alpha=0.35,
               linewidths=0, rasterized=True, label="ANN")
    lo, hi = view_limits(obs, test["wc_ww_ratio"].to_numpy(float))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(1.0, color="0.45", lw=0.9, ls=":")
    rmse_ann = float(np.sqrt(np.mean((test["wc_ww_ratio"] - obs) ** 2)))
    rmse_one = float(np.sqrt(np.mean((1.0 - obs) ** 2)))
    ax.text(0.04, 0.96,
            f"RMSE\n  ANN        {rmse_ann:.3f}\n  predict 1.0 {rmse_one:.3f}",
            transform=ax.transAxes, fontsize=11, va="top", family="monospace")
    ax.set_title("c) Against the constant baseline", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("USGS observed ratio [-]", fontsize=10)
    ax.set_ylabel("Predicted ratio [-]", fontsize=10)
    ax.legend(fontsize=9, loc="lower right", frameon=True)
    ax.grid(alpha=0.25, lw=0.5)

    if np.isfinite(s1["r2"]):
        print(f"  a) CV state 1960-1980   R2 {s1['r2']:+.4f}  "
              f"RMSE {s1['rmse']:.4f}  bias {s1['bias']:+.4f}  n {s1['n']:,}")
    elif np.isfinite(s1["rmse"]):
        print(f"  a) CV state 1960-1980   mean fold RMSE {s1['rmse']:.4f}  "
              f"n {s1['n']:,}   (per-fold bars; out-of-fold table absent)")
    print(f"  b) test county 1985-95  R2 {s2['r2']:+.4f}  RMSE {s2['rmse']:.4f}"
          f"  bias {s2['bias']:+.4f}  n {s2['n']:,}"
          f"  above 1.0: {s2['above_one']:,}")
    print(f"  c) predict-1.0 baseline RMSE {rmse_one:.4f} "
          f"vs ANN {rmse_ann:.4f}"
          f"  -> ANN is {'better' if rmse_ann < rmse_one else 'WORSE'}")

    fig.suptitle("Water-consumption to water-withdrawal ratio: downscaling "
                 "performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _show(fig)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scatter plots for the ratio downscaling ANN.",
        allow_abbrev=False)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--size", type=float, default=6.0)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")
    plot(args.data_dir, args.cmap, args.size)
    return 0


if __name__ == "__main__":
    # Not `raise SystemExit(main())`: under %run that aborts the cell before
    # the figure is flushed.
    main()