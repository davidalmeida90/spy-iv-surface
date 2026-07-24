"""Loader for the SPY EOD options dataset (Kaggle: dudesurfin/spy-options-eod-...).

The raw parquet is WIDE: one row per (quote_date, expiry, strike) carrying BOTH
the call and the put side, with bracketed column names like [C_IV], [P_IV],
[STRIKE], [UNDERLYING_LAST]. IV and Greeks are pre-computed by the dataset author.

This module melts that into a clean LONG format — one row per OTM option quote —
and applies the liquidity / sanity filters needed for a usable IV surface:

    OTM-only:   puts struck below spot, calls struck above spot
                (each side gives the cleaner IV away from deep-ITM intrinsic value)
    liquidity:  bid > 0, relative spread (ask-bid)/mid <= MAX_REL_SPREAD
    sanity:     IV within [IV_MIN, IV_MAX], drops negative / blow-up IV junk
    horizon:    DTE within [DTE_MIN, DTE_MAX], drops 0-DTE noise and sparse LEAPS

Output columns:
    date, expiry, dte, tau, spot, strike, type, mid, iv, moneyness, log_m
    (tau = time to expiry in years; named tau not T to avoid pandas' .T transpose)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "historical"

# --- filter thresholds (tweak here, they are the only knobs that matter) ---
DTE_MIN, DTE_MAX = 7, 365        # 1 week to 1 year — the liquid, dense part
IV_MIN, IV_MAX = 0.03, 2.00      # 3%..200% — kills negatives and 40.0 blow-ups
MAX_REL_SPREAD = 0.50            # drop quotes where (ask-bid)/mid > 50%


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().strip("[]") for c in df.columns]
    return df


def _side(df: pd.DataFrame, prefix: str, otm_mask: np.ndarray, opt_type: str) -> pd.DataFrame:
    """Extract one side (call or put) as long rows, OTM-filtered + cleaned."""
    bid = pd.to_numeric(df[f"{prefix}_BID"], errors="coerce")
    ask = pd.to_numeric(df[f"{prefix}_ASK"], errors="coerce")
    iv = pd.to_numeric(df[f"{prefix}_IV"], errors="coerce")
    mid = (bid + ask) / 2.0
    rel_spread = (ask - bid) / mid.replace(0, np.nan)

    out = pd.DataFrame({
        "date": pd.to_datetime(df["QUOTE_DATE"]),
        "expiry": pd.to_datetime(df["EXPIRE_DATE"]),
        "dte": pd.to_numeric(df["DTE"], errors="coerce"),
        "spot": pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce"),
        "strike": pd.to_numeric(df["STRIKE"], errors="coerce"),
        "type": opt_type,
        "mid": mid,
        "iv": iv,
        "rel_spread": rel_spread,
        "bid": bid,
    })
    keep = (
        otm_mask
        & out["bid"].gt(0)
        & out["mid"].gt(0)
        & out["iv"].between(IV_MIN, IV_MAX)
        & out["rel_spread"].le(MAX_REL_SPREAD)
        & out["dte"].between(DTE_MIN, DTE_MAX)
    )
    return out.loc[keep].copy()


def load_year(year: int, verbose: bool = True) -> pd.DataFrame:
    """Load one year's parquet, return clean long-format OTM quotes."""
    path = DATA_DIR / f"spy_eod_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run the Kaggle download first.")
    df = _clean_cols(pd.read_parquet(path))

    spot = pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce")
    strike = pd.to_numeric(df["STRIKE"], errors="coerce")
    otm_put = (strike < spot).to_numpy()       # puts below spot
    otm_call = (strike > spot).to_numpy()       # calls above spot

    puts = _side(df, "P", otm_put, "put")
    calls = _side(df, "C", otm_call, "call")
    long = pd.concat([puts, calls], ignore_index=True)

    long["tau"] = long["dte"] / 365.0
    long["moneyness"] = long["strike"] / long["spot"]
    long["log_m"] = np.log(long["moneyness"])
    long = long[["date", "expiry", "dte", "tau", "spot", "strike",
                 "type", "mid", "iv", "moneyness", "log_m"]]

    if verbose:
        raw = len(df) * 2
        print(f"  {year}: {len(long):>9,} clean OTM quotes "
              f"({len(long)/raw*100:4.1f}% of {raw:,} raw call+put rows)")
    return long


def load_range(years: range | list[int], verbose: bool = True) -> pd.DataFrame:
    """Concatenate clean quotes across a range of years."""
    parts = [load_year(y, verbose=verbose) for y in years]
    out = pd.concat(parts, ignore_index=True)
    return out


if __name__ == "__main__":
    # quick smoke test on one year
    df = load_year(2023)
    print(df.head())
    print(f"\nIV mean {df.iv.mean():.3f}, tau mean {df['tau'].mean():.3f}, "
          f"moneyness {df.moneyness.min():.2f}-{df.moneyness.max():.2f}")
