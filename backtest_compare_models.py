import pandas as pd
import numpy as np
from pathlib import Path
from math import log
from scipy.stats import chi2

RAW_DIR = Path("data/raw")
TAB_DIR = Path("results/tables")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_LEVELS = [0.95, 0.99]

# ---- Crisis windows (edit if you want) ----
CRISIS_WINDOWS = [
    ("2008-09-01", "2009-06-30"),  # GFC core
    ("2020-02-15", "2020-06-30"),  # COVID crash
]

# ---- Map dataset label -> raw CSV + rolling EVT file ----
DATASETS = {
    "S&P 500": {
        "raw": "S&P 500.csv",
        "rolling_evt": "rolling_evt_gpd_S&P 500.csv"
    },
    "NASDAQ": {
        "raw": "NASDAQ.csv",  
        "rolling_evt": "rolling_evt_gpd_NASDAQ.csv"
    },
    "DOW JONES": {
        "raw": "DOW JONES.csv", 
        "rolling_evt": "rolling_evt_gpd_DOW JONES.csv"
    },
    "CITIGROUP": {
        "raw": "CITIGROUP.csv",       
        "rolling_evt": "rolling_evt_gpd_CITIGROUP.csv"
    }
}



def get_price_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for key in ["adj close", "adjusted close", "close", "adjclose"]:
        if key in cols:
            return df.columns[cols.index(key)]
    return df.columns[1]


def load_losses(dataset_raw_file: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_raw_file)
    df["Date"] = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.to_datetime(df.iloc[:, 0])

    price_col = get_price_col(df)
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Price"]).sort_values("Date")

    df["LogRet"] = np.log(df["Price"] / df["Price"].shift(1))
    df = df.dropna(subset=["LogRet"])
    df["Loss"] = -df["LogRet"]  # positive = loss

    return df[["Date", "Loss"]].copy()


# ---- Backtest tests ----

def kupiec_pof_test(violations: np.ndarray, alpha: float) -> float:
    """
    Kupiec Proportion of Failures test.
    H0: violation probability = (1-alpha)
    Returns p-value (higher = model calibration plausible)
    """
    n = len(violations)
    x = int(violations.sum())
    p = 1 - alpha

    if x == 0:
        # if zero violations, likelihood under model is (1-p)^n
        # compute LR safely
        pi_hat = 0.0
    else:
        pi_hat = x / n

    # Avoid log(0)
    def safe_log(a):
        return -1e12 if a <= 0 else log(a)

    L0 = (n - x) * safe_log(1 - p) + x * safe_log(p)
    L1 = (n - x) * safe_log(1 - pi_hat) + x * safe_log(pi_hat)

    LR = -2 * (L0 - L1)
    pval = 1 - chi2.cdf(LR, df=1)
    return float(pval)


def christoffersen_independence_test(violations: np.ndarray) -> float:
    """
    Christoffersen independence test.
    H0: violations are independent (not clustered)
    Returns p-value (higher = independence plausible)
    """
    v = violations.astype(int)
    if len(v) < 2:
        return np.nan

    # transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        prev, cur = v[i-1], v[i]
        if prev == 0 and cur == 0: n00 += 1
        if prev == 0 and cur == 1: n01 += 1
        if prev == 1 and cur == 0: n10 += 1
        if prev == 1 and cur == 1: n11 += 1

    # probabilities
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    p01 = safe_div(n01, n00 + n01)
    p11 = safe_div(n11, n10 + n11)
    p1  = safe_div(n01 + n11, n00 + n01 + n10 + n11)

    # log-likelihoods
    def ll(n0, n1, p):
        if p <= 0: return -1e12 if n1 > 0 else 0.0
        if p >= 1: return -1e12 if n0 > 0 else 0.0
        return n0 * log(1 - p) + n1 * log(p)

    L0 = ll(n00 + n10, n01 + n11, p1)
    L1 = ll(n00, n01, p01) + ll(n10, n11, p11)

    LR = -2 * (L0 - L1)
    pval = 1 - chi2.cdf(LR, df=1)
    return float(pval)


def crisis_mask(dates: pd.Series) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for start, end in CRISIS_WINDOWS:
        m |= (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
    return m


# ---- Load baseline outputs ----
def load_baseline_table(model_name: str) -> pd.DataFrame:
    # You already generate these CSVs in earlier scripts.
    # Adjust filenames if yours differ.
    if model_name == "Normal":
        f = TAB_DIR / "normal_baseline_results.csv"
    elif model_name == "Student-t":
        f = TAB_DIR / "student_t_baseline_results.csv"
    else:
        raise ValueError("Unknown model")
    return pd.read_csv(f)


def main():
    # Baseline (static) VaR tables
    normal_tbl = load_baseline_table("Normal")
    t_tbl = load_baseline_table("Student-t")

    rows = []

    for ds_name, info in DATASETS.items():
        raw_path = RAW_DIR / info["raw"]
        roll_path = TAB_DIR / info["rolling_evt"]

        # Load losses
        loss_df = load_losses(raw_path)

        # Rolling EVT
        roll = pd.read_csv(roll_path)
        roll["Date"] = pd.to_datetime(roll["Date"])
        roll = roll.sort_values("Date")

        merged_evt = pd.merge(loss_df, roll[["Date", "VaR95", "VaR99"]], on="Date", how="inner")
        merged_evt["Crisis"] = crisis_mask(merged_evt["Date"])

        # Static baselines: use same VaR for all days
        for model_name, tbl in [("Normal", normal_tbl), ("Student-t", t_tbl)]:
            for alpha in CONF_LEVELS:
                var_val = float(tbl[(tbl["Dataset"] == ds_name) & (tbl["Confidence"] == alpha)]["VaR"].iloc[0])
                # Build implied VaR series
                tmp = loss_df.copy()
                tmp["VaR"] = var_val
                tmp["Crisis"] = crisis_mask(tmp["Date"])
                viol = (tmp["Loss"] > tmp["VaR"]).to_numpy()

                rows.append({
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Confidence": alpha,
                    "VaR": var_val,
                    "ViolationRate": viol.mean(),
                    "Kupiec_p": kupiec_pof_test(viol, alpha),
                    "Independence_p": christoffersen_independence_test(viol),
                    "ViolRate_Crisis": (tmp.loc[tmp["Crisis"], "Loss"] > tmp.loc[tmp["Crisis"], "VaR"]).mean(),
                    "ViolRate_NonCrisis": (tmp.loc[~tmp["Crisis"], "Loss"] > tmp.loc[~tmp["Crisis"], "VaR"]).mean(),
                    "N": len(tmp)
                })

        # Rolling EVT model
        for alpha in CONF_LEVELS:
            col = "VaR95" if alpha == 0.95 else "VaR99"
            viol = (merged_evt["Loss"] > merged_evt[col]).to_numpy()

            rows.append({
                "Dataset": ds_name,
                "Model": "EVT-GPD Rolling",
                "Confidence": alpha,
                "VaR": np.nan,
                "ViolationRate": viol.mean(),
                "Kupiec_p": kupiec_pof_test(viol, alpha),
                "Independence_p": christoffersen_independence_test(viol),
                "ViolRate_Crisis": (merged_evt.loc[merged_evt["Crisis"], "Loss"] > merged_evt.loc[merged_evt["Crisis"], col]).mean(),
                "ViolRate_NonCrisis": (merged_evt.loc[~merged_evt["Crisis"], "Loss"] > merged_evt.loc[~merged_evt["Crisis"], col]).mean(),
                "N": len(merged_evt)
            })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "model_backtest_comparison.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(out.sort_values(["Dataset", "Confidence", "Model"]).to_string(index=False))


if __name__ == "__main__":
    main()
