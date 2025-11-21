import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from erfa import ErfaWarning
from astropy.time import Time
from scipy.optimize import curve_fit
from datetime import datetime, timezone

# Ignore warnings from astropy about ERFA
warnings.filterwarnings("ignore", category=ErfaWarning)


CSV_PATH = "observations_20251116_051444-CCD.csv"
PERIOD = 5.36629 # days
JD0 = 2460991.18731207 # Derived epoch from find_epoch.py

# -------------------------------------------------------------------
# Setup and clean data
# -------------------------------------------------------------------
# Read in the CSV file
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Initial row count: {len(df)}")

# Clean and convert data types
df["JD"] = pd.to_numeric(df["jd"], errors="coerce")
df["Magnitude"] = pd.to_numeric(df["mag"], errors="coerce")
df["Band"] = df["band"].astype(str).str.strip().str.upper()

# Drop rows with missing JD or Magnitude
df = df.dropna(subset=["JD", "Magnitude", "Band"])

# Convert JD to UTC datetime
df["Date"] = Time(df["JD"].values, format="jd").to_datetime()
df["Date"] = pd.to_datetime(df["Date"], utc=True)

# Keep only certain band observations
keep_bands = ["JOHNSON V"]
df = df[df["Band"].isin(keep_bands)]

if len(df) == 0:
    raise ValueError("No data left after filtering")

# Only keep magnitudes between 3.4 and 4.4 since these are the values that the star reaches
df = df[(df["Magnitude"] >= 3.3) & (df["Magnitude"] <= 4.55)]
print(f"After filtering row count:: {len(df)}")
print(f"Data range from {df['Date'].min()} to {df['Date'].max()}")
print(f"Magnitude range from {df['Magnitude'].min()} to {df['Magnitude'].max()}")

# -------------------------------------------------------------------
# === Plot magnitude vs real date ===
plt.figure(figsize=(10, 5))
plt.scatter(df["Date"], df["Magnitude"], s=8)
plt.axvline(x=datetime.now(timezone.utc), color='gray', linestyle=':', linewidth=1.5, label='Now (UTC)')
plt.gca().invert_yaxis()
plt.xlabel("Date (UTC)")
plt.ylabel("Magnitude (V)")
plt.title("Observed Light Curve (unfolded)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------

# Epoch reference JD0 for phase calculation
df["Phase"] = ((df["JD"] - JD0) / PERIOD) % 1

# === Plot phase-folded light curve ===
plt.figure(figsize=(10, 5))
plt.scatter(df["Phase"], df["Magnitude"], s=8, alpha=0.6)
plt.scatter(df["Phase"] + 1, df["Magnitude"], s=8, alpha=0.6)  # repeat cycle
plt.gca().invert_yaxis()
plt.xlabel("Phase")
plt.ylabel("Magnitude (V)")
plt.title("Phase-Folded Light Curve of δ Cephei")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------

# --- Create Past + Future Prediction Plot ---
# Use the fitted Fourier model:
def fourier_series(phase, *coeffs):
    order = (len(coeffs) - 1) // 2
    result = coeffs[0]
    for i in range(1, order + 1):
        Ai = coeffs[2*i - 1]
        Bi = coeffs[2*i]
        result += Ai * np.cos(2*np.pi*i*phase) + Bi * np.sin(2*np.pi*i*phase)
    return result

# --- Create timeline ---
today = Time.now().jd
past_start = df["JD"].min()
future_end = today + 180   # predict 180 days into the future

# Prepare data for fitting
phase = df["Phase"].values
mag = df["Magnitude"].values

# Choose number of harmonics
order = 6   # Cepheids work well with 4–6

# Initial guess of parameters
ncoeffs = 1 + 2*order
p0 = np.zeros(ncoeffs)
p0[0] = np.median(mag)  # baseline magnitude
p0[1::2] = 0.1          # cosine terms initial guess
p0[2::2] = 0.0          # sine terms initial guess

# Fit Fourier series
print("Fitting Fourier series...")
params, _ = curve_fit(fourier_series, phase, mag, p0=p0, maxfev=30000)

print("Fourier fit coefficients:")
print(params)

# How far back and forward to plot
plot_past_days = 360
plot_future_days = 31

# Define time window
latest_jd = df["JD"].max()
start_jd = latest_jd - plot_past_days
end_jd = latest_jd + plot_future_days

# Select past data
df_recent = df[df["JD"] >= start_jd].copy()

# Generate time grid for model
jd_grid = np.linspace(start_jd, end_jd, 2000)
phase_grid = ((jd_grid - JD0) / PERIOD) % 1
mag_grid = fourier_series(phase_grid, *params)

time_grid = pd.to_datetime(Time(jd_grid, format="jd").to_datetime(), utc=True)

# Plot
plt.figure(figsize=(12, 6))

plt.scatter(df_recent["Date"], df_recent["Magnitude"], s=10, alpha=0.7, label="Recent CCD Observations")
plt.plot(time_grid, mag_grid, "r-", linewidth=2, label="Fourier Model Prediction")
plt.axvline(x=datetime.now(timezone.utc), color='gray', linestyle=':', linewidth=1.5, label='Now (UTC)')

plt.gca().invert_yaxis()
plt.title("Observed and Predicted Light Curve (Local Window)")
plt.xlabel("Date (UTC)")
plt.ylabel("Magnitude (V)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------

plot_past_days = 75
plot_future_days = 15

# Define time window
latest_jd = df["JD"].max()
start_jd = latest_jd - plot_past_days
end_jd = latest_jd + plot_future_days

# Select past data
df_recent = df[df["JD"] >= start_jd].copy()

# Generate time grid for model
jd_grid = np.linspace(start_jd, end_jd, 2000)
phase_grid = ((jd_grid - JD0) / PERIOD) % 1
mag_grid = fourier_series(phase_grid, *params)

time_grid = pd.to_datetime(Time(jd_grid, format="jd").to_datetime(), utc=True)

# Plot
plt.figure(figsize=(12, 6))

plt.scatter(df_recent["Date"], df_recent["Magnitude"], s=20, alpha=1, label="Recent CCD Observations")
plt.plot(time_grid, mag_grid, "r-", linewidth=1, label="Fourier Model Prediction")
plt.axvline(x=datetime.now(timezone.utc), color='gray', linestyle=':', linewidth=1.5, label='Now (UTC)')

plt.axhline(y=4.3700, color='gray', linestyle='--', linewidth=1.5, label='E')
plt.axhline(y=4.1475, color='gray', linestyle='--', linewidth=1.5, label='D')
plt.axhline(y=3.9250, color='gray', linestyle='--', linewidth=1.5, label='C')
plt.axhline(y=3.7025, color='gray', linestyle='--', linewidth=1.5, label='B')
plt.axhline(y=3.4800, color='gray', linestyle='--', linewidth=1.5, label='A')

df_own_data = [
    {"Date": pd.Timestamp("2025-09-03 22:10", tz="UTC"), "mag": 3.9250},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-09-10 21:05", tz="UTC"), "mag": 4.1475},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-09-11 22:32", tz="UTC"), "mag": 4.1475},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-09-15 23:34", tz="UTC"), "mag": 4.1475},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-09-22 22:02", tz="UTC"), "mag": 4.3700},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-09-25 23:10", tz="UTC"), "mag": 3.9250},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-10-02 21:02", tz="UTC"), "mag": 4.3700},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-10-07 20:55", tz="UTC"), "mag": 4.3700},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-10-08 20:49", tz="UTC"), "mag": 4.3700},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-10-23 22:16", tz="UTC"), "mag": 3.7025},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-11-04 23:12", tz="UTC"), "mag": 3.9250},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-11-11 22:51", tz="UTC"), "mag": 3.4800},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-11-13 22:12", tz="UTC"), "mag": 3.9250},  # Own observation (UTC)
    {"Date": pd.Timestamp("2025-11-15 23:13", tz="UTC"), "mag": 3.9250},  # Own observation (UTC)
]

plt.scatter(pd.to_datetime([d["Date"] for d in df_own_data], format="jd"), [d["mag"] for d in df_own_data], s=45, color="green", alpha=1.0, label="Own Observations")

plt.gca().invert_yaxis()
plt.title("CCD + Visually Observed and Predicted Light Curve (Local Window)")
plt.xlabel("Date (UTC)")
plt.ylabel("Magnitude (V)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
