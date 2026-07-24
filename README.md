# SPY Implied-Volatility Surface — Out-of-Time Deep Learning

Train two pure-numpy MLPs on **real** historical SPY EOD option chains (2010–2023)
to learn the implied-volatility surface, validated **out-of-time** (train on older
years, test on the years that came after).

This is the honest, multi-year version of the `nn_learning_smile` reel, which had
trained two small MLPs on a single live yfinance snapshot of a synthetic surface.

## Data

Kaggle: [`dudesurfin/spy-options-eod-volatility-surface-2010-2023`](https://www.kaggle.com/datasets/dudesurfin/spy-options-eod-volatility-surface-2010-2023)
— 14 years of EOD chains, one parquet per year (~0.6 GB), Greeks + IV pre-computed.

Download (one-time, needs a Kaggle API token in `.env` → `KAGGLE_USERNAME` / `KAGGLE_KEY`):

```bash
py -3 -m kaggle datasets download -d dudesurfin/spy-options-eod-volatility-surface-2010-2023 \
    -p ./data/historical --unzip
```

`data/` is gitignored (re-downloadable).

## Pipeline

| File | Role |
|---|---|
| `loader.py` | Wide→long melt; OTM-only filter (puts below spot, calls above); drop zero-bid / wide-spread / junk-IV / 0-DTE rows. Output: one clean quote per row. |
| `nets.py` | Two pure-numpy MLPs (Adam + ReLU), ported from the reel. Features `[x, τ, x², τ², x·τ]` where `x = ln(K/S)`. Target = total variance `w = σ²·τ`. |
| `train_iv_surface.py` | Chronological split, train both nets, report train/val/test IV-RMSE, plot loss curves + error-region heatmap. |

## Architectures

| Net | Layers | Params |
|---|---|---|
| SMALL | `5-4-4-1` | 49 |
| BIG | `5-128-128-1` | 17,409 |

## Out-of-time split

```
TRAIN   2010–2020   (11 years)
VAL     2021–2022   (2 years, unseen)
TEST    2023        (1 year, unseen — reported once)
```

Purely chronological, no shuffling across the boundary → VAL/TEST measure genuine
generalization to future regimes.

## Run

```bash
py -3 train_iv_surface.py
```

Outputs `outputs/loss_curves.png` (the train-vs-val overfitting story) and
`outputs/error_regions.png` (where the fit broke on 2023, by moneyness × maturity).

## Surfaces

A few of the learned surfaces are in `videos/` as short clips: the network
prediction (warm colors) climbing into the empirical market surface (the white
web) over training. Files are named `surface_<arch>_<activation>_<features>.mp4`,
so `surface_deep_tanh_wing.mp4` is the strongest model and
`surface_deep_relu_wing.mp4` shows the same size with ReLU for contrast.

## Result plots

`results/` holds the figures behind the tables below: `loss_curves.png`
(train vs validation), `error_regions.png` (where the fit breaks by moneyness and
maturity), `feature_comparison.png`, `model_comparison.png`, and `skew_error.png`.

## Results

Out-of-time IV RMSE (trained on 400k subsampled OTM quotes from 2010–2020):

| model | params | train | val (2021–22) | test (2023) |
|---|---|---|---|---|
| SMALL | 49 | 6.65% | 5.21% | 3.32% |
| BIG | 17,409 | 6.32% | 4.84% | 3.35% |

**What the network learned, and where it broke:**

1. **A 49-param net already captures the surface.** Going from 49 → 17,409
   params improves out-of-sample IV RMSE by only ~0.3–0.4 pp. The SPY IV surface
   is smooth in (log-moneyness, τ), so capacity barely matters — the opposite of
   the engineered overfitting in the original reel (which forced it with noise +
   ~1k points).
2. **No overfitting — and val/test error is *lower* than train.** With a decade
   and 400k points, neither net memorizes. Train error is *higher* than val/test
   because TRAIN (2010–2020) includes the 2020 COVID vol explosion — genuinely
   harder, jagged surfaces — while 2021–2023 were calmer. The remaining error is
   regime difficulty, not generalization failure.
3. **The error lives in the short-dated far-OTM call wing** (τ < 0.17,
   K/S > 1.10): ~7–8% there vs ~2% on the long-dated body. That steep upper-right
   corner of the smile is the genuinely hard region (see `outputs/error_regions.png`).

### Experiment 2 — macro/regime features (negative result)

Hypothesis: conditioning on market state (realized vol, VIX, rates, spreads from
FRED) should close the out-of-time gap. It **did the opposite.**

| feature set | net | train | val | test 2023 |
|---|---|---|---|---|
| geometric (5) | SMALL | 6.86% | 5.47% | 3.46% |
| **geometric (5)** | **BIG** | 6.32% | 4.92% | **3.19% ← best** |
| +macro no-VIX (9) | SMALL | 4.61% | 6.16% | 9.76% |
| +macro no-VIX (9) | BIG | 3.25% | 5.87% | 6.02% |
| +macro +VIX (10) | SMALL | 2.73% | 3.82% | 4.15% |
| +macro +VIX (10) | BIG | 2.34% | 5.41% | 5.60% |

Every macro variant fits **train** better but generalizes **worse** — the signature
of **out-of-distribution extrapolation (covariate shift).** The 3-month T-bill was
~0% through most of 2010–2020 (ZIRP) but ~5.3% in 2023, so the standardized rate
feature in the test year sits far outside the training support, where ReLU nets
extrapolate badly. The geometric features survive because moneyness/maturity are
*stationary* (same meaning every regime); macro *levels* are not. Two sub-findings:
VIX partially rescues the damage (range-bound + directly proxies IV level, but
semi-circular and still loses to baseline); and for the first time **more capacity =
worse** (BIG overfits the training regime harder than SMALL). Lesson: don't feed raw
non-stationary levels — use stationary transforms (VIX percentile, rate *changes*) or
isolate realized vol (which is more stationary than rates). Code: `macro.py`,
feature sets in `train_iv_surface.FEATURE_SETS`. Plot: `outputs/feature_comparison.png`.

### Experiment 3 — model comparison & robustness to the rate covariate shift

Harder split (TRAIN 2010-2018 / VAL 2019-2020 / TEST 2021-2023; 3m rate 0-2.45% in
train vs 0-5.63% in test). 5 models x 3 feature sets. `compare_models.py`,
`outputs/model_comparison.png`.

TEST 2021-2023 IV RMSE (%):

| model | geometric | +macro no-VIX | +macro +VIX | penalty |
|---|---|---|---|---|
| MLP-ReLU | 5.37 | **12.10** | 7.38 | **+2.01** |
| MLP-tanh | 5.45 | 8.32 | 4.32 | −1.14 |
| MLP-L2 | 5.84 | 6.42 | 4.26 | −1.58 |
| RandomForest | 5.52 | 5.16 | 3.75 | −1.77 |
| XGBoost | 5.29 | 5.11 | **3.51** | −1.78 |

**The ReLU MLP is the only model that breaks when macro is added** — every other
model benefits. The robustness ladder is monotonic in how each handles out-of-range
inputs: ReLU extrapolates linearly (blows up) -> tanh saturates -> L2 shrinks reliance
-> trees clamp at the training-range boundary (bounded error). VIX helps every model
(in-distribution, proxies the IV level). **Selection lesson:** XGBoost wins on raw test
(3.51%) but MLP-L2 wins on *validation* (val 4.32 / test 4.26) — pick by validation, not
test. Caveat: val (2019-20) contains COVID and is a harder regime than test, so chrono
val-selection is imperfect — an argument for the random split (cleaner i.i.d. val/test).

### Experiment 4 — random split vs chronological (the covariate-shift proof)

Pooled all 5.87M quotes 2010-2023, random 400k/200k/200k split, same 5 models x 3
feature sets. `random_split_compare.py`, `outputs/random_split_results.npy`.

Random-split TEST IV RMSE (%):

| model | geometric | +macro no-VIX | +macro +VIX | penalty |
|---|---|---|---|---|
| MLP-ReLU | 5.67 | 3.34 | 2.45 | −3.22 |
| MLP-tanh | 5.68 | 2.87 | 1.79 | −3.89 |
| MLP-L2 | 5.89 | 3.44 | 2.42 | −3.47 |
| RandomForest | 5.65 | 2.58 | 1.86 | −3.79 |
| XGBoost | 5.63 | 1.32 | **1.01** | **−4.62** |

**Total reversal of Experiment 3.** Every macro penalty flips from positive (hurts,
chronological) to strongly negative (helps, random). The decisive cell: ReLU + no-VIX
macro = **12.10% chronological vs 3.34% random** — an ~8.8pp swing from the split alone,
same model + features. Two structural fingerprints of the random split: (1) geometric
features score the same both ways (stationary, no shift to suffer); (2) train≈val≈test to
two decimals everywhere (i.i.d., zero generalization gap, vs chronological's 4/10/5.4).

**Conclusion:** the macro features are genuinely *informative* (halve the error when there
is no shift) but *unusable out-of-time* (extrapolate catastrophically). The entire
"macro hurts" effect is temporal covariate shift, not the features. **Caveat for using a
random split going forward:** its low numbers (XGBoost+VIX = 1.01%) are INTERPOLATION skill,
not forward-deployable generalization — the same config degrades to 3.5%+ (ReLU: 12%) out of
time. Use random for the smoothing/interpolation goal; never read its macro-boosted numbers
as evidence the features help in production.

### Honest-modeling note

The first run trained on **total variance** `w = σ²·τ` (per the spec's optional
suggestion). That inverted the capacity ranking — the BIG net looked *worse* in
IV space — purely as an artifact: reporting error via `IV = √(w/τ)` divides by τ,
blowing up tiny `w`-errors at short maturities that the MSE-on-`w` objective
barely weighted. Switching the target to **raw IV** (so objective = metric) fixed
it. See the comment in `nets.make_target`.
