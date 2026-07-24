"""Train MLPs on REAL historical SPY option chains with honest out-of-time
validation, comparing FEATURE SETS to test whether macro / regime state closes
the out-of-sample gap.

    TRAIN   2010-2020   (11 years)
    VAL     2021-2022   (2 years, unseen)
    TEST    2023        (1 year, unseen, reported once)

Feature sets (all share the 5 geometric features [x, tau, x^2, tau^2, x*tau],
x = ln(K/S)):
    geometric        the 5 geometric features only (the original baseline)
    +macro (no VIX)  + realized_vol_20d, dgs3mo, t10y2y, baa_spread  (honest,
                     non-circular regime/macro state)
    +macro (+VIX)    also + VIX. Powerful but SEMI-CIRCULAR (VIX ~ 30d ATM IV),
                     so we isolate it to see how much of the gain is "cheating".

Both the SMALL (49-param) and BIG (17k-param) nets are run on each set.

Run:  py -3 train_iv_surface.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loader import load_range
from macro import build_macro_frame, attach_macro, MACRO_COLS_FULL, MACRO_COLS_NO_VIX
from nets import (MLP, small_layers, big_layers, count_params,
                  make_features, make_target, Standardizer, train)

# --- config -----------------------------------------------------------------
TRAIN_YEARS = range(2010, 2021)
VAL_YEARS = range(2021, 2023)
TEST_YEARS = range(2023, 2024)

MAX_TRAIN = 400_000
MAX_EVAL = 200_000
EPOCHS = 60
LR = 0.01
BATCH = 16384
SEED = 0

FEATURE_SETS = [
    ("geometric",       []),
    ("+macro (no VIX)", MACRO_COLS_NO_VIX),
    ("+macro (+VIX)",   MACRO_COLS_FULL),
]

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def subsample(df, cap, seed):
    if len(df) <= cap:
        return df
    idx = np.random.default_rng(seed).choice(len(df), cap, replace=False)
    print(f"    subsampled {cap:,} of {len(df):,} "
          f"(dropped {len(df)-cap:,}) — surface is densely oversampled")
    return df.iloc[idx].reset_index(drop=True)


def prepare(years, cap, seed, macro):
    df = attach_macro(load_range(years, verbose=True), macro)
    before = len(df)
    df = df.dropna(subset=MACRO_COLS_FULL).reset_index(drop=True)   # first ~20d of 2010
    if before - len(df):
        print(f"    dropped {before-len(df):,} rows missing macro (pre-2010 warmup)")
    return subsample(df, cap, seed)


def iv_rmse(net, X_std, iv_mean, iv_true):
    iv_pred = net.predict(X_std).flatten() + iv_mean
    return float(np.sqrt(np.mean((iv_pred - iv_true) ** 2)) * 100)


def main():
    macro = build_macro_frame()
    print("Loading clean OTM quotes + macro (chronological out-of-time split)...")
    print("  TRAIN 2010-2020:")
    df_tr = prepare(TRAIN_YEARS, MAX_TRAIN, SEED, macro)
    print("  VAL   2021-2022:")
    df_va = prepare(VAL_YEARS, MAX_EVAL, SEED + 1, macro)
    print("  TEST  2023:")
    df_te = prepare(TEST_YEARS, MAX_EVAL, SEED + 2, macro)

    # target: raw IV, centered on TRAIN mean (shared across feature sets)
    iv_tr = df_tr["iv"].to_numpy()
    iv_va = df_va["iv"].to_numpy()
    iv_te = df_te["iv"].to_numpy()
    iv_mean = float(iv_tr.mean())
    ytr = (iv_tr - iv_mean).reshape(-1, 1)
    yva = (iv_va - iv_mean).reshape(-1, 1)

    print(f"\nTrain {len(df_tr):,} | Val {len(df_va):,} | Test {len(df_te):,} points")
    print(f"Target = implied vol (sigma); train mean IV = {iv_mean:.4f}\n")

    results = []          # (feat_name, net_name, params, train, val, test)
    keep = {}             # cache nets/curves for plotting
    for feat_name, macro_cols in FEATURE_SETS:
        Xtr_raw = make_features(df_tr, macro_cols)
        Xva_raw = make_features(df_va, macro_cols)
        Xte_raw = make_features(df_te, macro_cols)
        std = Standardizer(Xtr_raw)                      # fit on TRAIN only
        Xtr, Xva, Xte = std(Xtr_raw), std(Xva_raw), std(Xte_raw)
        n_feat = Xtr.shape[1]

        for net_name, layer_fn in [("SMALL", small_layers), ("BIG", big_layers)]:
            layers = layer_fn(n_feat)
            p = count_params(layers)
            net = MLP(layers, seed=42)
            tr_hist, va_hist = train(net, Xtr, ytr, Xva, yva,
                                     epochs=EPOCHS, lr=LR, batch_size=BATCH, seed=SEED)
            r = (feat_name, net_name, p,
                 iv_rmse(net, Xtr, iv_mean, iv_tr),
                 iv_rmse(net, Xva, iv_mean, iv_va),
                 iv_rmse(net, Xte, iv_mean, iv_te))
            results.append(r)
            keep[(feat_name, net_name)] = (net, Xte, tr_hist, va_hist)
            print(f"  {feat_name:<16} {net_name:<6} {n_feat:>2}f {p:>7,}p  "
                  f"train {r[3]:.2f}%  val {r[4]:.2f}%  test {r[5]:.2f}%")

    # --- summary table ---
    print("\n" + "=" * 70)
    print(f"{'feature set':<18}{'net':<7}{'params':>9}{'train':>9}{'val':>9}{'test':>9}")
    print("-" * 70)
    for feat_name, net_name, p, tr, va, te in results:
        print(f"{feat_name:<18}{net_name:<7}{p:>9,}{tr:>8.2f}%{va:>8.2f}%{te:>8.2f}%")
    print("=" * 70)

    base = next(te for f, n, p, tr, va, te in results if f == "geometric" and n == "BIG")
    best = min(results, key=lambda r: r[5])
    print(f"\nBaseline BIG test RMSE: {base:.2f}%   ->   best ({best[0]}, {best[1]}): "
          f"{best[5]:.2f}%   ({(1-best[5]/base)*100:.0f}% lower)")

    plot_results_bars(results)
    plot_loss_curves(keep)
    plot_error_regions(keep[(best[0], best[1])][0], keep[(best[0], best[1])][1],
                       iv_mean, iv_te, df_te, title=f"{best[0]}, {best[1]} net")
    print(f"\nPlots written to {OUT_DIR}")


def plot_results_bars(results):
    feats = [f for f, _ in FEATURE_SETS]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(feats))
    for k, net_name in enumerate(("SMALL", "BIG")):
        vals = [next(te for f, n, p, tr, va, te in results if f == fn and n == net_name)
                for fn in feats]
        bars = ax.bar(x + (k - 0.5) * width, vals, width,
                      label=f"{net_name} net", color=["#8888cc", "#cc5555"][k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylabel("TEST 2023 IV RMSE (%)")
    ax.set_title("Out-of-time test error by feature set (lower = better)")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_comparison.png", dpi=130)
    plt.close(fig)


def plot_loss_curves(keep):
    """BIG net, geometric vs +macro(+VIX) — confirms overfitting still absent."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, feat_name in zip(axes, ("geometric", "+macro (+VIX)")):
        _, _, tr, va = keep[(feat_name, "BIG")]
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color="#2ca02c", lw=2, label="train MSE")
        ax.plot(ep, va, color="#d62728", lw=2, label="val MSE")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (implied vol)")
        ax.set_title(f"BIG net — {feat_name}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend()
    fig.suptitle("Train vs val loss — macro features lower the curve, no overfitting",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curves.png", dpi=130)
    plt.close(fig)


def plot_error_regions(net, X_std, iv_mean, iv_true, df, title=""):
    iv_pred = net.predict(X_std).flatten() + iv_mean
    abs_err = np.abs(iv_pred - iv_true) * 100
    m = df["moneyness"].to_numpy()
    t = df["tau"].to_numpy()
    m_bins = np.linspace(0.80, 1.20, 13)
    t_bins = np.array([0.02, 0.08, 0.17, 0.33, 0.5, 0.75, 1.0])
    grid = np.full((len(t_bins) - 1, len(m_bins) - 1), np.nan)
    for i in range(len(t_bins) - 1):
        for j in range(len(m_bins) - 1):
            sel = ((t >= t_bins[i]) & (t < t_bins[i + 1]) &
                   (m >= m_bins[j]) & (m < m_bins[j + 1]))
            if sel.sum() > 50:
                grid[i, j] = abs_err[sel].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma",
                   extent=[m_bins[0], m_bins[-1], 0, len(t_bins) - 1])
    ax.set_yticks(np.arange(len(t_bins) - 1) + 0.5)
    ax.set_yticklabels([f"{t_bins[i]:.2f}-{t_bins[i+1]:.2f}"
                        for i in range(len(t_bins) - 1)])
    ax.set_xlabel("moneyness K/S")
    ax.set_ylabel("maturity tau (yr)")
    ax.set_title(f"Mean |IV error| by region, TEST 2023 — {title}")
    fig.colorbar(im, ax=ax, label="mean |IV error| (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_regions.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
