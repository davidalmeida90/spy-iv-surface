"""Decompose the dashboard models' TRAIN RMSE by (moneyness x maturity) region,
to see WHERE the error lives and which region contributes most of the total.

Replicates build_viz_data's exact data + training (random split, same seeds, macro),
then bins the per-quote prediction error.

Run:  py -3 analyze_rmse_regions.py
"""
from __future__ import annotations
import numpy as np
from loader import load_range
from macro import build_macro_frame, attach_macro, MACRO_COLS_EXT
from nets import MLP, make_features, Standardizer

N_TRAIN, SEED, EPOCHS, LR, BATCH = 80_000, 0, 100, 0.01, 8192

print("Loading data (same random split as the dashboard)...")
macro = build_macro_frame()
df = attach_macro(load_range(range(2010, 2024), verbose=False), macro)
df = df.dropna(subset=MACRO_COLS_EXT).reset_index(drop=True)
idx = np.random.default_rng(SEED).permutation(len(df))
df_tr = df.iloc[idx[:N_TRAIN]].reset_index(drop=True)

std = Standardizer(make_features(df_tr, MACRO_COLS_EXT))
Xtr = std(make_features(df_tr, MACRO_COLS_EXT))
iv_tr = df_tr["iv"].to_numpy()
iv_mean = float(iv_tr.mean())
ytr = (iv_tr - iv_mean).reshape(-1, 1)
m = df_tr["moneyness"].to_numpy()
t = df_tr["tau"].to_numpy()

MB = [("<0.85", 0, 0.85), ("0.85-0.95", 0.85, 0.95), ("ATM .95-1.05", 0.95, 1.05),
      ("1.05-1.15", 1.05, 1.15), (">1.15", 1.15, 9)]
TB = [("<0.08", 0, 0.08), ("0.08-0.25", 0.08, 0.25), ("0.25-0.5", 0.25, 0.5), (">0.5", 0.5, 9)]


def train(layers, act):
    net = MLP(layers, seed=42, activation=act)
    rng = np.random.default_rng(SEED)
    n = Xtr.shape[0]
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            _, a, z = net.forward(Xtr[b]); net.backward(ytr[b], a, z, LR)
    return net


for name, layers, act in [("Small ReLU", [18, 4, 4, 1], "relu"),
                          ("Big tanh", [18, 128, 128, 1], "tanh")]:
    net = train(layers, act)
    pred = net.predict(Xtr).flatten() + iv_mean
    err = pred - iv_tr
    sse_total = float(np.sum(err ** 2))
    overall = np.sqrt(np.mean(err ** 2)) * 100
    print(f"\n{'='*78}\n{name}:  overall train RMSE = {overall:.2f}%   (n={len(err):,})\n{'='*78}")

    # --- RMSE (%) per region ---
    print("RMSE (%) by region:")
    print(f"{'maturity':<12}" + "".join(f"{mb[0]:>14}" for mb in MB))
    for tlab, tlo, thi in TB:
        row = f"{tlab:<12}"
        for mlab, mlo, mhi in MB:
            sel = (t >= tlo) & (t < thi) & (m >= mlo) & (m < mhi)
            row += (f"{np.sqrt(np.mean(err[sel]**2))*100:>13.2f}%" if sel.sum() > 30 else f"{'-':>14}")
        print(row)

    # --- share of TOTAL squared error (where the RMSE 'comes from') ---
    print("\nShare of total squared error (%):")
    print(f"{'maturity':<12}" + "".join(f"{mb[0]:>14}" for mb in MB))
    for tlab, tlo, thi in TB:
        row = f"{tlab:<12}"
        for mlab, mlo, mhi in MB:
            sel = (t >= tlo) & (t < thi) & (m >= mlo) & (m < mhi)
            share = float(np.sum(err[sel]**2)) / sse_total * 100
            row += f"{share:>13.1f}%"
        print(row)
    # short-dated OTM-call corner specifically
    wing = (t < 0.08) & (m > 1.05)
    print(f"\n  short-dated OTM calls (T<0.08, K/S>1.05): "
          f"{wing.mean()*100:.1f}% of points, but {np.sum(err[wing]**2)/sse_total*100:.1f}% of total error, "
          f"RMSE {np.sqrt(np.mean(err[wing]**2))*100:.2f}%")
