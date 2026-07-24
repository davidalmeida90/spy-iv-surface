"""Where on the smile does the error live? Breaks the champion model's TEST error
down by skew region (deep OTM put -> ATM -> deep OTM call) x maturity.

Uses the headline champion: geometric features, BIG ReLU net, original split
(train 2010-2020, test 2023). Produces a region RMSE table + an
"error across the smile" line plot (RMSE vs moneyness, one line per maturity bucket).

Run:  py -3 skew_breakdown.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loader import load_range
from nets import MLP, big_layers, make_features, make_target, Standardizer, train

OUT_DIR = Path(__file__).parent / "outputs"

# skew regions by moneyness m = K/S
REGIONS = [
    ("deep OTM put",  0.00, 0.90),
    ("OTM put",       0.90, 0.97),
    ("ATM",           0.97, 1.03),
    ("OTM call",      1.03, 1.10),
    ("deep OTM call", 1.10, 9.99),
]
# maturity buckets (years)
MATS = [("short <2m", 0.0, 0.17), ("med 2-6m", 0.17, 0.5), ("long >6m", 0.5, 9.9)]


def rmse_pct(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) * 100) if len(t) else np.nan


def main():
    print("Training champion (geometric BIG ReLU, train 2010-2020)...")
    df_tr = load_range(range(2010, 2021), verbose=False).sample(
        n=400_000, random_state=0).reset_index(drop=True)
    df_te = load_range(range(2023, 2024), verbose=False).sample(
        n=200_000, random_state=2).reset_index(drop=True)

    Xtr_raw = make_features(df_tr)
    std = Standardizer(Xtr_raw)
    Xtr, Xte = std(Xtr_raw), std(make_features(df_te))
    iv_tr = df_tr["iv"].to_numpy()
    iv_te = df_te["iv"].to_numpy()
    iv_mean = float(iv_tr.mean())
    ytr = (iv_tr - iv_mean).reshape(-1, 1)
    yte = (iv_te - iv_mean).reshape(-1, 1)

    net = MLP(big_layers(Xtr.shape[1]), seed=42)
    train(net, Xtr, ytr, Xte, yte, epochs=60, lr=0.01, batch_size=16384, seed=0)
    pred = net.predict(Xte).flatten() + iv_mean
    err = pred - iv_te
    m = df_te["moneyness"].to_numpy()
    t = df_te["tau"].to_numpy()

    print(f"\nOverall TEST IV RMSE: {rmse_pct(pred, iv_te):.2f}%\n")

    # --- region x maturity table ---
    header = f"{'region':<15}" + "".join(f"{mat[0]:>12}" for mat in MATS) + f"{'ALL':>10}"
    print(header)
    print("-" * len(header))
    for rname, mlo, mhi in REGIONS:
        rmask = (m >= mlo) & (m < mhi)
        row = f"{rname:<15}"
        for _, tlo, thi in MATS:
            sel = rmask & (t >= tlo) & (t < thi)
            row += f"{rmse_pct(pred[sel], iv_te[sel]):>11.2f}%"
        row += f"{rmse_pct(pred[rmask], iv_te[rmask]):>9.2f}%"
        print(row)
    # bias check: signed mean error per region (over/under-prediction)
    print("\nSigned mean error (pred - actual, IV %), all maturities:")
    for rname, mlo, mhi in REGIONS:
        rmask = (m >= mlo) & (m < mhi)
        print(f"  {rname:<15} {err[rmask].mean()*100:+6.2f}%   (n={rmask.sum():,})")

    plot_smile_error(m, t, np.abs(err) * 100)
    print(f"\nPlot written to {OUT_DIR / 'skew_error.png'}")


def plot_smile_error(m, t, abs_err):
    bins = np.linspace(0.80, 1.20, 17)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mat_name, tlo, thi in MATS:
        tmask = (t >= tlo) & (t < thi)
        ys = []
        for j in range(len(bins) - 1):
            sel = tmask & (m >= bins[j]) & (m < bins[j + 1])
            ys.append(abs_err[sel].mean() if sel.sum() > 30 else np.nan)
        ax.plot(centers, ys, marker="o", lw=2, label=mat_name)
    ax.axvspan(0.97, 1.03, color="gray", alpha=0.12, label="ATM")
    ax.axvline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("moneyness K/S   (<1 = OTM puts,  >1 = OTM calls)")
    ax.set_ylabel("mean |IV error| (%)")
    ax.set_title("Error across the smile — TEST 2023 (champion: geometric BIG ReLU)")
    ax.legend(title="maturity")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "skew_error.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
