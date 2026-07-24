"""Model comparison under a HARDER, longer-horizon out-of-time split, focused on
one question: which model families survive the interest-rate covariate shift?

    TRAIN 2010-2018   (9 yrs; max 3m T-bill ~2.4%)
    VAL   2019-2020   (COVID vol lands here)
    TEST  2021-2023   (3 yrs; rates spike to ~5.3% -> far outside train range)

Models:
    MLP-ReLU    our baseline net (extrapolates linearly outside training support)
    MLP-tanh    bounded activation -> saturates instead of exploding
    MLP-L2      ReLU + weight decay -> shrinks reliance on shifting features
    RandomForest  trees CLAMP at the training-range boundary (no extrapolation)
    XGBoost       boosted trees, same clamping property

Each model is run on 3 feature sets (geometric / +macro no-VIX / +macro +VIX) so the
*penalty* from adding the non-stationary macro features is visible per model family.
The prediction: trees pay a far smaller penalty than the ReLU net.

Run:  py -3 compare_models.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from loader import load_range
from macro import build_macro_frame, attach_macro, MACRO_COLS_FULL, MACRO_COLS_NO_VIX
from nets import MLP, big_layers, make_features, Standardizer, train

TRAIN_YEARS = range(2010, 2019)     # 2010-2018
VAL_YEARS = range(2019, 2021)       # 2019-2020
TEST_YEARS = range(2021, 2024)      # 2021-2023

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
MODELS = ["MLP-ReLU", "MLP-tanh", "MLP-L2", "RandomForest", "XGBoost"]

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def subsample(df, cap, seed):
    if len(df) <= cap:
        return df
    idx = np.random.default_rng(seed).choice(len(df), cap, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def prepare(years, cap, seed, macro):
    df = attach_macro(load_range(years, verbose=False), macro)
    df = df.dropna(subset=MACRO_COLS_FULL).reset_index(drop=True)
    return subsample(df, cap, seed)


def rmse_pct(pred_iv, true_iv):
    return float(np.sqrt(np.mean((pred_iv - true_iv) ** 2)) * 100)


def fit_predict(model_name, Xtr, Xva, Xte, iv_tr, iv_va, iv_mean, ytr_c, yva_c):
    """Return (train_pred, val_pred, test_pred) in IV units for the given model."""
    n = Xtr.shape[1]
    if model_name in ("MLP-ReLU", "MLP-tanh", "MLP-L2"):
        act = "tanh" if model_name == "MLP-tanh" else "relu"
        l2 = 1e-3 if model_name == "MLP-L2" else 0.0
        net = MLP(big_layers(n), seed=42, activation=act, l2=l2)
        train(net, Xtr, ytr_c, Xva, yva_c, epochs=EPOCHS, lr=LR, batch_size=BATCH, seed=SEED)
        f = lambda X: net.predict(X).flatten() + iv_mean
    elif model_name == "RandomForest":
        rf = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=50,
                                   max_samples=0.5, n_jobs=-1, random_state=0)
        rf.fit(Xtr, iv_tr)
        f = rf.predict
    elif model_name == "XGBoost":
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           n_jobs=-1, random_state=0)
        xgb.fit(Xtr, iv_tr)
        f = xgb.predict
    else:
        raise ValueError(model_name)
    return f(Xtr), f(Xva), f(Xte)


def main():
    macro = build_macro_frame()
    print("Split: TRAIN 2010-2018 | VAL 2019-2020 | TEST 2021-2023\n")
    df_tr = prepare(TRAIN_YEARS, MAX_TRAIN, SEED, macro)
    df_va = prepare(VAL_YEARS, MAX_EVAL, SEED + 1, macro)
    df_te = prepare(TEST_YEARS, MAX_EVAL, SEED + 2, macro)

    iv_tr = df_tr["iv"].to_numpy()
    iv_va = df_va["iv"].to_numpy()
    iv_te = df_te["iv"].to_numpy()
    iv_mean = float(iv_tr.mean())
    ytr_c = (iv_tr - iv_mean).reshape(-1, 1)
    yva_c = (iv_va - iv_mean).reshape(-1, 1)

    # report the rate extrapolation explicitly
    print(f"Train 3m-rate range: {df_tr['dgs3mo'].min():.2f}-{df_tr['dgs3mo'].max():.2f}%"
          f"   Test 3m-rate range: {df_te['dgs3mo'].min():.2f}-{df_te['dgs3mo'].max():.2f}%")
    print(f"Train {len(df_tr):,} | Val {len(df_va):,} | Test {len(df_te):,}\n")

    results = {}          # (feat, model) -> (train, val, test)
    for feat_name, macro_cols in FEATURE_SETS:
        Xtr_raw = make_features(df_tr, macro_cols)
        std = Standardizer(Xtr_raw)
        Xtr = std(Xtr_raw)
        Xva = std(make_features(df_va, macro_cols))
        Xte = std(make_features(df_te, macro_cols))
        for model_name in MODELS:
            ptr, pva, pte = fit_predict(model_name, Xtr, Xva, Xte,
                                        iv_tr, iv_va, iv_mean, ytr_c, yva_c)
            r = (rmse_pct(ptr, iv_tr), rmse_pct(pva, iv_va), rmse_pct(pte, iv_te))
            results[(feat_name, model_name)] = r
            print(f"  {feat_name:<16} {model_name:<13} "
                  f"train {r[0]:5.2f}%  val {r[1]:5.2f}%  test {r[2]:5.2f}%")

    # --- table: test RMSE, model x feature set, with the macro penalty ---
    print("\n" + "=" * 78)
    print(f"TEST 2021-2023 IV RMSE (%)   [penalty = +macro(+VIX) minus geometric]")
    print(f"{'model':<14}{'geometric':>12}{'+macro noVIX':>15}{'+macro +VIX':>14}{'penalty':>10}")
    print("-" * 78)
    for model_name in MODELS:
        g = results[("geometric", model_name)][2]
        nv = results[("+macro (no VIX)", model_name)][2]
        fv = results[("+macro (+VIX)", model_name)][2]
        print(f"{model_name:<14}{g:>11.2f}%{nv:>14.2f}%{fv:>13.2f}%{fv-g:>+9.2f}")
    print("=" * 78)

    plot_bars(results)
    print(f"\nPlot written to {OUT_DIR / 'model_comparison.png'}")


def plot_bars(results):
    feats = [f for f, _ in FEATURE_SETS]
    colors = {"geometric": "#4c72b0", "+macro (no VIX)": "#dd8452", "+macro (+VIX)": "#c44e52"}
    x = np.arange(len(MODELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, feat_name in enumerate(feats):
        vals = [results[(feat_name, m)][2] for m in MODELS]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=feat_name,
                      color=colors[feat_name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("TEST 2021-2023 IV RMSE (%)")
    ax.set_title("Robustness to the rate covariate shift — test error by model x feature set\n"
                 "(tall orange/red bars = a model that breaks when macro is added)")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
