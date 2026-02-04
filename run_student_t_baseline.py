import pandas as pd
import numpy as np
from scipy.stats import t
from pathlib import Path

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF_LEVELS = [0.95, 0.99]


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def student_t_var_es(returns, alpha):
    df, loc, scale = t.fit(returns)

    var = -(loc + scale * t.ppf(1 - alpha, df))
    es = scale * (t.pdf(t.ppf(1 - alpha, df), df) /
                  ((1 - alpha))) * ((df + t.ppf(1 - alpha, df) ** 2) / (df - 1)) - loc

    return var, es, df


results = []

for csv_file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)
    prices = df.iloc[:, 1]
    returns = compute_log_returns(prices)

    for alpha in CONF_LEVELS:
        var, es, df_param = student_t_var_es(returns, alpha)

        results.append({
            "Dataset": csv_file.stem,
            "Model": "Student-t",
            "Confidence": alpha,
            "VaR": var,
            "ES": es,
            "Degrees_of_Freedom": df_param
        })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "student_t_baseline_results.csv", index=False)

print(results_df)
