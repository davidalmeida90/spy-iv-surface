"""Two pure-numpy MLPs (Adam + ReLU) that learn the SPY implied-vol surface.

The MLP / Adam / feature machinery is ported almost verbatim from the reel
    Social_media_automation/experimentation_machine/nn_learning_smile.py
which fit a SYNTHETIC surface on ~1k grid points with full-batch gradient
descent. Here we fit REAL multi-year option chains (hundreds of thousands of
points), so the only substantive change is mini-batch training.

Architectures (param counts match the reel exactly):
    SMALL  [5, 4, 4, 1]      ->     49 params   (capacity-limited)
    BIG    [5, 128, 128, 1]  -> 17,409 params   (plenty of capacity)

Features (5), from log-moneyness x = ln(K/S) and time-to-expiry tau:
    [x, tau, x^2, tau^2, x*tau]

Target: total variance  w = sigma^2 * tau  (smoother & better-scaled than raw
IV; we convert predictions back to IV = sqrt(w/tau) for reporting).
"""
from __future__ import annotations
import math
import numpy as np

SMALL_LAYERS = [5, 4, 4, 1]
BIG_LAYERS = [5, 128, 128, 1]


def small_layers(n_features: int):
    return [n_features, 4, 4, 1]


def big_layers(n_features: int):
    return [n_features, 128, 128, 1]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def make_features(df, macro_cols=(), smile=False, curve=False, vix=False) -> np.ndarray:
    """Polynomial features in (log-moneyness, tau), optionally + the SMILE
    COORDINATE (x/sqrt(T), 1/sqrt(T)), the short-T CURVATURE terms (x^2/T, x/T),
    and the VIX d1 coordinate x/(sigma*sqrt(T)) with sigma=VIX/100 (the TRUE
    standardized moneyness, using VIX as the sigma proxy), then + macro columns."""
    x = df["log_m"].to_numpy()
    tau = df["tau"].to_numpy()
    feats = [x, tau, x ** 2, tau ** 2, x * tau]
    if smile:
        rt = np.sqrt(np.maximum(tau, 1e-6))
        feats += [x / rt, 1.0 / rt]                    # standardized-moneyness + 1/sqrt(T)
    if curve:
        it = 1.0 / np.maximum(tau, 1e-6)
        feats += [x ** 2 * it, x * it]                 # smile & skew over maturity (1/T blow-up)
    if vix:
        rt = np.sqrt(np.maximum(tau, 1e-6))
        sigma = np.maximum(df["vix"].to_numpy() / 100.0, 1e-3)   # VIX% -> decimal sigma proxy
        feats += [x / (sigma * rt)]                    # d1-like: log(K/S) / (sigma*sqrt(T))
    for c in macro_cols:
        feats.append(df[c].to_numpy())
    return np.stack(feats, axis=1)


def make_target(df) -> np.ndarray:
    """Raw implied vol sigma, shape (n, 1).

    We tried total variance w = sigma^2 * tau (the textbook-smoother target), but
    reporting error in IV space then needs IV = sqrt(w/tau): the 1/tau divide
    blows up tiny w-errors at short maturities, AND the MSE-on-w objective barely
    weights those points. Net effect: the IV-space ranking inverted (the big net
    looked worse) purely as a reconstruction artifact. Training on raw IV makes
    the optimization objective match the IV-RMSE metric, so the comparison is honest.
    """
    return df["iv"].to_numpy().reshape(-1, 1)


class Standardizer:
    """Fit mean/std on TRAIN only; apply to val/test (no leakage)."""
    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-9

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def count_params(layers) -> int:
    return sum(layers[i] * layers[i + 1] + layers[i + 1]
              for i in range(len(layers) - 1))


# ---------------------------------------------------------------------------
# MLP — Adam optimizer, ReLU hidden, linear output (ported from the reel)
# ---------------------------------------------------------------------------
class MLP:
    """Adam MLP. activation='relu' or 'tanh'; l2 = weight-decay coefficient.

    tanh and L2 are here specifically to probe robustness to out-of-distribution
    inputs: a tanh net saturates (bounded extrapolation) where ReLU extrapolates
    linearly to infinity, and L2 shrinks reliance on any single shifting feature.
    """
    def __init__(self, sizes, seed=42, activation="relu", l2=0.0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.activation = activation
        self.l2 = l2
        self.W, self.b = [], []
        self.mW, self.mb, self.vW, self.vb = [], [], [], []
        for i in range(len(sizes) - 1):
            # He init for relu/softplus, Xavier for tanh
            std = (math.sqrt(2.0 / sizes[i]) if activation in ("relu", "softplus")
                   else math.sqrt(1.0 / sizes[i]))
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * std)
            self.b.append(np.zeros((1, sizes[i + 1])))
            self.mW.append(np.zeros_like(self.W[-1]))
            self.mb.append(np.zeros_like(self.b[-1]))
            self.vW.append(np.zeros_like(self.W[-1]))
            self.vb.append(np.zeros_like(self.b[-1]))
        self.t = 0

    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "softplus":
            return np.logaddexp(0.0, z)                # smooth: log(1 + e^z)
        return np.tanh(z)

    def _act_deriv(self, z):
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        if self.activation == "softplus":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))   # sigmoid
        return 1.0 - np.tanh(z) ** 2

    def forward(self, X):
        activations = [X]
        z_values = []
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            z_values.append(z)
            a = self._act(z) if i < len(self.W) - 1 else z
            activations.append(a)
        return a, activations, z_values

    def predict(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = self._act(z) if i < len(self.W) - 1 else z
        return a

    def backward(self, y, activations, z_values, lr,
                 beta1=0.9, beta2=0.999, eps=1e-8, weights=None):
        self.t += 1
        m = y.shape[0]
        if weights is None:
            delta = 2 * (activations[-1] - y) / m
        else:                                          # weighted MSE (upweight tails)
            delta = 2 * weights * (activations[-1] - y) / weights.sum()
        for i in reversed(range(len(self.W))):
            a_prev = activations[i]
            dW = a_prev.T @ delta + self.l2 * self.W[i]      # weight decay
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._act_deriv(z_values[i - 1])
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * dW * dW
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * db * db
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def train(net: MLP, X, y, X_val, y_val, epochs, lr, batch_size=8192, seed=0):
    """Mini-batch Adam. Returns per-epoch (train_mse, val_mse) on the centered
    target. Loss is full-set MSE measured once per epoch for a clean curve."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    train_hist, val_hist = [], []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            _, acts, zs = net.forward(X[idx])
            net.backward(y[idx], acts, zs, lr)
        train_hist.append(float(np.mean((net.predict(X) - y) ** 2)))
        val_hist.append(float(np.mean((net.predict(X_val) - y_val) ** 2)))
    return np.array(train_hist), np.array(val_hist)
