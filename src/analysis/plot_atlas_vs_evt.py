import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EVT_FILE = Path("results/tables/rolling_evt_gpd_S&P 500.csv")
ATLAS_FILE = Path("results/tables/atlas_v1_S&P 500.csv")

evt = pd.read_csv(EVT_FILE)
atlas = pd.read_csv(ATLAS_FILE)

evt["Date"] = pd.to_datetime(evt["Date"])
atlas["Date"] = pd.to_datetime(atlas["Date"])

# align on dates
df = pd.merge(evt[["Date", "VaR99"]], atlas[["Date", "VaR99", "q_t"]], on="Date", suffixes=("_EVT", "_ATLAS"))

plt.figure()
plt.plot(df["Date"], df["VaR99_EVT"], label="Rolling EVT VaR99")
plt.plot(df["Date"], df["VaR99_ATLAS"], label="ATLAS v1 VaR99")
plt.title("ATLAS v1 vs Rolling EVT (VaR99)")
plt.xlabel("Date")
plt.ylabel("VaR99 (loss)")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df["Date"], df["q_t"])
plt.title("ATLAS v1 Adaptive Threshold Quantile q(t)")
plt.xlabel("Date")
plt.ylabel("q(t)")
plt.tight_layout()
plt.show()