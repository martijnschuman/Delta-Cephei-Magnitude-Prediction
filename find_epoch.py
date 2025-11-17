import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from erfa import ErfaWarning
from astropy.time import Time
from datetime import datetime, timezone

# Ignore ERFA warnings
warnings.filterwarnings("ignore", category=ErfaWarning)

CSV_PATH = "observations_20251116_051444-CCD.csv"

# === Load CSV ===
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Columns:", df.columns)
print("Initial rows:", len(df))

# === Clean data ===
df["JD"] = pd.to_numeric(df["jd"], errors="coerce")
df["Magnitude"] = pd.to_numeric(df["mag"], errors="coerce")
df["Band"] = df["band"].astype(str).str.strip().str.upper()

# Drop rows with missing JD or Magnitude
df = df.dropna(subset=["JD", "Magnitude", "Band"])

# === Convert JD → datetime (not strictly needed, but nice) ===
df["Date"] = Time(df["JD"].values, format="jd").to_datetime()
df["Date"] = pd.to_datetime(df["Date"], utc=True)

# Keep only certain band observations
keep_bands = ["JOHNSON V"]
df = df[df["Band"].isin(keep_bands)]

# Only keep magnitudes between 3.4 and 4.4 since these are the values that the star reaches
df = df[(df["Magnitude"] >= 3.3) & (df["Magnitude"] <= 4.55)]
print(f"After filtering row count:: {len(df)}")
print(f"Data range from {df['Date'].min()} to {df['Date'].max()}")
print(f"Magnitude range from {df['Magnitude'].min()} to {df['Magnitude'].max()}")

# === Known Cepheid period ===
PERIOD = 5.36629

# === Compute phase ===
jd_ref = np.median(df["JD"].values)  # arbitrary but helps fitting stability
phase = ((df["JD"].values - jd_ref) / PERIOD) % 1
mag = df["Magnitude"].values

# === Fourier series model ===
order = 5  # 5 harmonics = good for Cepheids
def fourier_series(phase, *coeffs):
    res = np.full_like(phase, coeffs[0])
    for i in range(order):
        A = coeffs[1 + 2*i]
        B = coeffs[2 + 2*i]
        res += A * np.cos(2*np.pi*(i+1)*phase) + B * np.sin(2*np.pi*(i+1)*phase)
    return res

ncoeffs = 1 + 2*order
p0 = np.zeros(ncoeffs)
p0[0] = np.median(mag)
p0[1::2] = 0.1  # initial cosine guesses

print("Fitting Fourier model...")
params, _ = curve_fit(fourier_series, phase, mag, p0=p0, maxfev=30000)

# === Find phase of maximum brightness ===
phase_grid = np.linspace(0, 1, 4000)
fit_vals = fourier_series(phase_grid, *params)
phase_of_max = phase_grid[np.argmin(fit_vals)]  # min mag = max brightness

# === Convert phase → JD0 ===
jd0_raw = jd_ref + phase_of_max * PERIOD

# Move JD0 to the cycle nearest the latest CCD observation
max_jd = df["JD"].max()
ncycles = round((max_jd - jd0_raw) / PERIOD)
JD0 = jd0_raw + ncycles * PERIOD

print("\n===============================")
print(" Derived Epoch (JD0):", JD0)
print("===============================\n")
