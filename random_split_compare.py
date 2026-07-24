"""RANDOM split vs the chronological out-of-time split — the canonical
demonstration of why random splits flatter time-series models.

Pools ALL quotes 2010-2023, shuffles, and draws disjoint 400k/200k/200k
train/val/test sets at random (so train/val/test are i.i.d. from the same
distribution). Runs the same 5 models x 3 feature sets as compare_models.py.

Two predictions to check against the chronological numbers:
  1. Random test RMSE should be MUCH lower (no regime gap -> the model
     interpolates among neighbors it effectively already saw).
  2. The macro penalty should vanish or reverse: with 2023's high rates also in
     TRAIN, the rate feature is no longer an out-of-distribution extrapolation.

Run:  py -3 random_split_compare.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loader import load_range
from macro import build_macro_frame, attach_macro, MACRO_COLS_FULL
from nets import make_features, Standardizer
from compare_models import fit_predict, FEATURE_SETS, MODELS, rmse_pct, EPOCHS  # reuse

ALL_YEARS = range(2010, 2024)
N_TRAIN, N_EVAL = 400_000, 200_000
SEED = 7
OUT_DIR = Path(__file__).parent / "outputs"


def main():
    macro = build_macro_frame()
    print("Pooling ALL quotes 2010-2023 and splitting RANDOMLY...")
    df = attach_macro(load_range(ALL_YEARS, verbose=False), macro)
    df = df.dropna(subset=MACRO_COLS_FULL).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    tr_idx = idx[:N_TRAIN]
    va_idx = idx[N_TRAIN:N_TRAIN + N_EVAL]
    te_idx = idx[N_TRAIN + N_EVAL:N_TRAIN + 2 * N_EVAL]
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    iv_tr = df_tr["iv"].to_numpy()
    iv_va = df_va["iv"].to_numpy()
    iv_te = df_te["iv"].to_numpy()
    iv_mean = float(iv_tr.mean())
    ytr_c = (iv_tr - iv_mean).reshape(-1, 1)
    yva_c = (iv_va - iv_mean).reshape(-1, 1)

    print(f"Random split from a pool of {len(df):,} quotes "
          f"(train {N_TRAIN:,} | val {N_EVAL:,} | test {N_EVAL:,}); "
          f"test now spans every year, not just the future.\n")

    results = {}
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

    print("\n" + "=" * 78)
    print("RANDOM-SPLIT TEST IV RMSE (%)   [penalty = +macro(+VIX) minus geometric]")
    print(f"{'model':<14}{'geometric':>12}{'+macro noVIX':>15}{'+macro +VIX':>14}{'penalty':>10}")
    print("-" * 78)
    for model_name in MODELS:
        g = results[("geometric", model_name)][2]
        nv = results[("+macro (no VIX)", model_name)][2]
        fv = results[("+macro (+VIX)", model_name)][2]
        print(f"{model_name:<14}{g:>11.2f}%{nv:>14.2f}%{fv:>13.2f}%{fv-g:>+9.2f}")
    print("=" * 78)
    print("\nCompare these to the chronological numbers from compare_models.py:")
    print("  - if random test RMSE << chronological, the time split was the hard part")
    print("  - if the macro penalty is ~0 / negative here, it was covariate shift")

    np.save(OUT_DIR / "random_split_results.npy", results, allow_pickle=True)
    print(f"\nResults saved to {OUT_DIR / 'random_split_results.npy'}")


if __name__ == "__main__":
    main()
