import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import genpareto

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF_LEVELS = [0.95, 0.99]
THRESHOLD_Q = 0.95  # start simple: 95th percentile of LOSSES


def get_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Robustly pick the price column.
    Tries common column names; otherwise falls back to 2nd column.
    """
    cols = [c.lower() for c in df.columns]
    for key in ["adj close", "adjusted close", "close", "adjclose"]:
        if key in cols:
            return df[df.columns[cols.index(key)]]
    return df.iloc[:, 1]


def compute_log_returns(prices: pd.Series) -> np.ndarray:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    r = np.log(prices / prices.shift(1)).dropna()
    return r.to_numpy()


def evt_var_es_from_gpd(losses: np.ndarray, alpha: float, u: float, xi: float, beta: float) -> tuple[float, float]:
    """
    losses: array of positive losses (L = -returns)
    alpha: confidence level (0.95 or 0.99)
    u: threshold on losses
    xi, beta: GPD shape and scale fitted to exceedances y = L-u, L>u

    POT tail formula:
    P(L > x) = p_u * (1 + xi*(x-u)/beta)^(-1/xi), for x>u
    where p_u = P(L>u) ≈ k/n
    """
    n = len(losses)
    exceed = losses[losses > u]
    k = len(exceed)
    if k < 30:
        raise ValueError(f"Too few exceedances ({k}). Increase data length or lower threshold.")

    p_u = k / n
    tail_prob = 1 - alpha  # e.g., 0.01 for 99%

    if xi == 0:
        var = u + beta * np.log(p_u / tail_prob)
    else:
        var = u + (beta / xi) * ((p_u / tail_prob) ** xi - 1)

    # ES exists only if xi < 1
    if xi >= 1:
        es = np.nan
    else:
        y0 = var - u  # exceedance level corresponding to VaR
        mean_excess_over_y0 = (beta + xi * y0) / (1 - xi)
        es = var + mean_excess_over_y0

    return var, es


results = []

for csv_file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)

    prices = get_price_series(df)
    returns = compute_log_returns(prices)

    # losses = positive numbers for negative returns
    losses = -returns
    losses = losses[np.isfinite(losses)]

    # threshold u from loss distribution
    u = np.quantile(losses, THRESHOLD_Q)

    exceedances = losses[losses > u] - u

    # Fit GPD to exceedances with location fixed at 0 (standard POT)
    xi, loc, beta = genpareto.fit(exceedances, floc=0)

    for alpha in CONF_LEVELS:
        var, es = evt_var_es_from_gpd(losses, alpha, u, xi, beta)

        results.append({
            "Dataset": csv_file.stem,
            "Model": "EVT-GPD",
            "Threshold_q": THRESHOLD_Q,
            "Threshold_u": u,
            "Exceedances_k": int((losses > u).sum()),
            "N": int(len(losses)),
            "Confidence": alpha,
            "VaR": float(var),
            "ES": float(es) if np.isfinite(es) else np.nan,
            "xi_shape": float(xi),
            "beta_scale": float(beta),
        })

results_df = pd.DataFrame(results)
out_path = RESULTS_DIR / "evt_gpd_baseline_results.csv"
results_df.to_csv(out_path, index=False)

print(results_df)
print(f"\nSaved: {out_path}")