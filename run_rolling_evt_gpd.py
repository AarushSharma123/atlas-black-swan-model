import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import genpareto

DATA_DIR = Path("data/raw")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000          # ~4 years of trading days
THRESH_Q = 0.95        # top 5% losses
CONF_LEVELS = [0.95, 0.99]
MIN_EXCEED = 50        # safety: skip windows with too few exceedances


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

    # ES exists only for xi < 1
    if xi >= 1:
        es = np.nan
    else:
        y0 = var - u
        es = var + (beta + xi * y0) / (1 - xi)

    return float(var), float(es)


def rolling_evt_for_file(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # Date parsing (stooq usually has Date column)
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])

    prices = get_price_series(df)
    returns = compute_log_returns(prices)

    # Align dates to returns (returns drops first row)
    dates = dates.iloc[-len(returns):].reset_index(drop=True)
    returns = returns.reset_index(drop=True)

    # Losses: positive values represent negative returns
    losses_all = (-returns).to_numpy()

    rows = []

    for end_idx in range(WINDOW, len(losses_all)):
        window_losses = losses_all[end_idx - WINDOW:end_idx]
        window_date = dates.iloc[end_idx]  # end-of-window date

        # threshold in THIS window
        u = np.quantile(window_losses, THRESH_Q)

        exceedances = window_losses[window_losses > u] - u
        k = len(exceedances)

        if k < MIN_EXCEED:
            continue

        # Fit GPD to exceedances, loc fixed at 0
        xi, loc, beta = genpareto.fit(exceedances, floc=0)

        # Compute rolling EVT risk metrics
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
            "Threshold_q": THRESH_Q,
            "Threshold_u": float(u),
            "Exceedances_k": int(k),
            "xi_shape": float(xi),
            "beta_scale": float(beta),
            **metrics
        })

    return pd.DataFrame(rows)


def main():
    for csv_file in DATA_DIR.glob("*.csv"):
        out = rolling_evt_for_file(csv_file)
        out_path = OUT_DIR / f"rolling_evt_gpd_{csv_file.stem}.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved {csv_file.stem}: {out_path}  (rows={len(out)})")


if __name__ == "__main__":
    main()