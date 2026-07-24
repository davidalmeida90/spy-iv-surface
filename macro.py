"""Macro / regime features to condition the IV-surface model on market state.

Our diagnosis was that the dominant error came from training ONE static surface
across all of 2010-2023 while the real surface's *level* drifts with the vol
regime. These features give the network that missing state:

    realized_vol_20d   trailing 20-day realized vol of SPY (annualized).
                       Computed from the dataset's own daily closes -> HONEST,
                       not circular: it is backward-looking realized movement,
                       independent of option prices.
    vix                CBOE VIX (FRED VIXCLS). Powerful but SEMI-CIRCULAR: VIX is
                       itself ~30-day ATM implied vol of SPX, so it is close to
                       "the answer" for the IV level. We include it but report
                       results with and without it so its contribution is visible.
    dgs3mo             3-month T-bill yield (FRED) — short rate / cost of carry.
    t10y2y             10y-2y term spread (FRED) — yield-curve slope / cycle.
    baa_spread         Moody's BAA corporate yield minus 10y Treasury
                       (FRED BAA10Y) — credit spread / risk appetite. (Chosen over
                       the HY OAS series, which FRED only serves back to 2023.)

All series are daily, contemporaneous (as-of the quote date, never future), and
forward-filled across holidays. Cached to data/macro_cache.csv after first build.
"""
from __future__ import annotations
from pathlib import Path
import urllib.request
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "data" / "macro_cache.csv"

FRED_SERIES = {
    "vix": "VIXCLS",
    "vix3m": "VXVCLS",        # 3-month vol (FRED has no VIX3M) -> used for the ratio
    "dgs3mo": "DGS3MO",
    "dgs2": "DGS2",
    "dgs5": "DGS5",
    "dgs10": "DGS10",
    "dgs30": "DGS30",
    "t10y2y": "T10Y2Y",
    "t10y3m": "T10Y3M",
    "baa_spread": "BAA10Y",
    "nfci": "NFCI",           # Chicago Fed financial conditions (weekly)
    "t5yifr": "T5YIFR",       # 5y5y forward inflation breakeven
}

# feature column groups
# original 5-feature sets (keep for the experiment scripts / documented results)
MACRO_COLS_FULL = ["realized_vol_20d", "vix", "dgs3mo", "t10y2y", "baa_spread"]
MACRO_COLS_NO_VIX = ["realized_vol_20d", "dgs3mo", "t10y2y", "baa_spread"]
# extended set: vol term-structure + full curve + financial conditions + inflation
MACRO_COLS_EXT = ["realized_vol_20d", "vix", "vix_ts",
                  "dgs3mo", "dgs2", "dgs5", "dgs10", "dgs30",
                  "t10y2y", "t10y3m", "baa_spread", "nfci", "t5yifr"]


def _load_env_key() -> str:
    env = HERE / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("FRED_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("FRED_API_KEY not found in .env")


def _fetch_fred(series_id: str, key: str) -> pd.DataFrame:
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id="
           f"{series_id}&api_key={key}&file_type=json&observation_start=2009-06-01")
    obs = json.load(urllib.request.urlopen(url, timeout=30))["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # "." -> NaN
    return df.set_index("date")["value"]


def _realized_vol_from_spy() -> pd.Series:
    """Trailing 20-day annualized realized vol from the dataset's daily closes.

    Reads only [QUOTE_DATE] + [UNDERLYING_LAST] from each yearly parquet (fast),
    dedups to one close per day, and rolls a 20d std over the full 2010-2023
    series so the trailing window is continuous across year boundaries.
    """
    data_dir = HERE / "data" / "historical"
    spots = []
    for path in sorted(data_dir.glob("spy_eod_*.parquet")):
        d = pd.read_parquet(path, columns=["[QUOTE_DATE]", "[UNDERLYING_LAST]"])
        d.columns = ["date", "spot"]
        d["date"] = pd.to_datetime(d["date"])
        spots.append(d.groupby("date")["spot"].first())
    s = pd.concat(spots).sort_index()
    s = s[~s.index.duplicated()]
    logret = np.log(s / s.shift(1))
    rv = logret.rolling(20).std() * np.sqrt(252)
    rv.name = "realized_vol_20d"
    return rv


def build_macro_frame(rebuild: bool = False) -> pd.DataFrame:
    """Daily macro feature frame indexed by date, forward-filled. Cached to CSV."""
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, parse_dates=["date"]).set_index("date")

    key = _load_env_key()
    cols = {name: _fetch_fred(sid, key) for name, sid in FRED_SERIES.items()}
    macro = pd.DataFrame(cols)
    macro["realized_vol_20d"] = _realized_vol_from_spy()
    # daily index spanning the data, forward-fill weekends/holidays/missing prints
    full_idx = pd.date_range(macro.index.min(), macro.index.max(), freq="D")
    macro = macro.reindex(full_idx).ffill()
    macro["vix_ts"] = macro["vix3m"] / macro["vix"]     # vol term-structure ratio
    macro.index.name = "date"
    macro = macro[MACRO_COLS_EXT]                        # keep the extended feature set
    macro.to_csv(CACHE)
    return macro


def attach_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro features onto option quotes by quote date."""
    out = df.merge(macro, left_on="date", right_index=True, how="left")
    return out


if __name__ == "__main__":
    m = build_macro_frame(rebuild=True)
    print(f"Macro frame: {m.shape[0]:,} days, cols = {list(m.columns)}")
    print(m.loc["2023-06-14":"2023-06-16"].to_string())
    print(f"\nNaN counts:\n{m.isna().sum()}")
