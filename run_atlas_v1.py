import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import genpareto

DATA_DIR = Path("data/raw")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000
CONF_LEVELS = [0.95, 0.99]
MIN_EXCEED = 50

# --- ATLAS v1 regime settings ---
VOL_WINDOW = 60  # rolling volatility lookback
# If volatility percentile is high, lower threshold quantile to get more tail points earlier
Q_CALM = 0.95
Q_ELEVATED = 0.92
Q_CRISIS = 0.90

# Volatility percentile cutoffs (based on history up to time t)
P_ELEVATED = 0.70  # 70th percentile
P_CRISIS = 0.90    # 90th percentile


def get_price_series(df: pd.DataFrame) -> pd.Series:
    cols = [c.lower() for c in df.columns]
    for key in ["adj close", "adjusted close", "close", "adjclose"]:
        if key in cols:
            return df[df.columns[cols.index(key)]]
    return df.iloc[:, 1]


def compute_log_returns(prices: pd.Series) -> pd.Series:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    r = np.log(prices / prices.shift(1)).dropna()
    return r


def evt_var_es(losses: np.ndarray, alpha: float, u: float, xi: float, beta: float) -> tuple[float, float]:
    n = len(losses)
    exceed = losses[losses > u]
    k = len(exceed)
    p_u = k / n
    tail_prob = 1 - alpha

    if xi == 0:
        var = u + beta * np.log(p_u / tail_prob)
    else:
        var = u + (beta / xi) * ((p_u / tail_prob) ** xi - 1)

    if xi >= 1:
        es = np.nan
    else:
        y0 = var - u
        es = var + (beta + xi * y0) / (1 - xi)

    return float(var), float(es)


def rolling_volatility(returns: pd.Series, w: int) -> pd.Series:
    # std of returns over lookback
    return returns.rolling(w).std()


def vol_percentile_series(vol: pd.Series) -> pd.Series:
    """
    At each time t, compute the percentile rank of vol(t) relative to vol history up to t.
    This avoids looking into the future.
    """
    out = np.full(len(vol), np.nan)
    vals = vol.to_numpy()

    for i in range(len(vals)):
        if not np.isfinite(vals[i]):
            continue
        hist = vals[: i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 30:
            continue
        out[i] = (hist <= vals[i]).mean()
    return pd.Series(out, index=vol.index)


def choose_q(vol_pct: float) -> float:
    if not np.isfinite(vol_pct):
        return Q_CALM
    if vol_pct >= P_CRISIS:
        return Q_CRISIS
    if vol_pct >= P_ELEVATED:
        return Q_ELEVATED
    return Q_CALM


def atlas_for_file(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # Date handling
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])

    prices = get_price_series(df)
    returns = compute_log_returns(prices)

    # Align dates to returns
    dates = dates.iloc[-len(returns):].reset_index(drop=True)
    returns = returns.reset_index(drop=True)

    losses_all = (-returns).to_numpy()

    # Volatility regime signal
    vol = rolling_volatility(returns, VOL_WINDOW)
    vol_pct = vol_percentile_series(vol).reset_index(drop=True)

    rows = []

    for end_idx in range(WINDOW, len(losses_all)):
        window_losses = losses_all[end_idx - WINDOW:end_idx]
        window_date = dates.iloc[end_idx]

        # adaptive threshold quantile based on vol percentile at time end_idx
        q_t = choose_q(vol_pct.iloc[end_idx])
        u = np.quantile(window_losses, q_t)

        exceedances = window_losses[window_losses > u] - u
        k = len(exceedances)
        if k < MIN_EXCEED:
            # If threshold becomes too strict in some windows, skip
            continue

        xi, loc, beta = genpareto.fit(exceedances, floc=0)

        metrics = {"VaR95": np.nan, "ES95": np.nan, "VaR99": np.nan, "ES99": np.nan}
        for alpha in CONF_LEVELS:
            var, es = evt_var_es(window_losses, alpha, u, xi, beta)
            if alpha == 0.95:
                metrics["VaR95"] = var
                metrics["ES95"] = es
            else:
                metrics["VaR99"] = var
                metrics["ES99"] = es

        rows.append({
            "Date": window_date,
            "Window": WINDOW,
            "VolWindow": VOL_WINDOW,
            "Vol": float(vol.iloc[end_idx]) if np.isfinite(vol.iloc[end_idx]) else np.nan,
            "VolPct": float(vol_pct.iloc[end_idx]) if np.isfinite(vol_pct.iloc[end_idx]) else np.nan,
            "q_t": float(q_t),
            "Threshold_u": float(u),
            "Exceedances_k": int(k),
            "xi_shape": float(xi),
            "beta_scale": float(beta),
            **metrics
        })

    return pd.DataFrame(rows)


def main():
    for csv_file in DATA_DIR.glob("*.csv"):
        out = atlas_for_file(csv_file)
        out_path = OUT_DIR / f"atlas_v1_{csv_file.stem}.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved {csv_file.stem}: {out_path} (rows={len(out)})")


if __name__ == "__main__":
    main()