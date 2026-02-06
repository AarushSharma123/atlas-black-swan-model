import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RAW_FILE = Path("data/raw/S&P 500.csv")
ROLL_FILE = Path("results/tables/rolling_evt_gpd_S&P 500.csv")  # your rolling output

# ---- Load raw prices ----
raw = pd.read_csv(RAW_FILE)
raw["Date"] = pd.to_datetime(raw["Date"])

# choose correct price column
price_col = None
for c in raw.columns:
    if c.lower() in ["adj close", "adjusted close", "close"]:
        price_col = c
        break
if price_col is None:
    price_col = raw.columns[1]

raw["Price"] = pd.to_numeric(raw[price_col], errors="coerce")
raw = raw.dropna(subset=["Price"]).sort_values("Date")

# log returns -> losses
raw["LogRet"] = np.log(raw["Price"] / raw["Price"].shift(1))
raw = raw.dropna(subset=["LogRet"])
raw["Loss"] = -raw["LogRet"]  # positive = loss

# ---- Load rolling EVT table ----
roll = pd.read_csv(ROLL_FILE)
roll["Date"] = pd.to_datetime(roll["Date"])
roll = roll.sort_values("Date")

# align by date (inner join so dates match)
df = pd.merge(raw[["Date", "Loss"]], roll[["Date", "VaR99", "VaR95"]], on="Date", how="inner")

# violations
df["Viol99"] = df["Loss"] > df["VaR99"]
df["Viol95"] = df["Loss"] > df["VaR95"]

# ---- Plot ----
plt.figure()
plt.plot(df["Date"], df["Loss"], label="Actual Daily Loss")
plt.plot(df["Date"], df["VaR99"], label="Rolling EVT VaR99")
plt.title("Actual Loss vs Rolling EVT VaR99 (S&P 500)")
plt.xlabel("Date")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# ---- Print violation rates ----
viol_rate_99 = df["Viol99"].mean()
viol_rate_95 = df["Viol95"].mean()

print("Violation rate VaR99:", round(viol_rate_99, 4), "(target ~0.01)")
print("Violation rate VaR95:", round(viol_rate_95, 4), "(target ~0.05)")
print("Total points:", len(df))
print("VaR99 violations:", int(df["Viol99"].sum()))
print("VaR95 violations:", int(df["Viol95"].sum()))