import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLE_DIR = Path("results/tables")

FILE = TABLE_DIR / "rolling_evt_gpd_S&P 500.csv"  # change if needed

df = pd.read_csv(FILE)
df["Date"] = pd.to_datetime(df["Date"])

# ---- Plot 1: xi over time ----
plt.figure()
plt.plot(df["Date"], df["xi_shape"])
plt.title("Rolling EVT Tail Shape (xi) Over Time")
plt.xlabel("Date")
plt.ylabel("xi (shape)")
plt.tight_layout()

# ---- Plot 2: VaR99 over time ----
plt.figure()
plt.plot(df["Date"], df["VaR99"])
plt.title("Rolling EVT VaR (99%) Over Time")
plt.xlabel("Date")
plt.ylabel("VaR99 (loss)")
plt.tight_layout()

# Show everything at once
plt.show()