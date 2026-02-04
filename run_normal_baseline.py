import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF_LEVELS = [0.95, 0.99]


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def normal_var_es(returns, alpha):
    mu = returns.mean()
    sigma = returns.std()

    z = norm.ppf(1 - alpha)
    var = -(mu + sigma * z)
    es = sigma * norm.pdf(z) / (1 - alpha) - mu

    return var, es


results = []

for csv_file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)

    prices = df.iloc[:, 1]  # adjusted close column
    returns = compute_log_returns(prices)

    for alpha in CONF_LEVELS:
        var, es = normal_var_es(returns, alpha)

        results.append({
            "Dataset": csv_file.stem,
            "Model": "Normal",
            "Confidence": alpha,
            "VaR": var,
            "ES": es
        })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "normal_baseline_results.csv", index=False)

print(results_df)