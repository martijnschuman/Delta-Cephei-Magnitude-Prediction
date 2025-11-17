import warnings
import pandas as pd
import matplotlib.pyplot as plt
from erfa import ErfaWarning
from astropy.time import Time
from datetime import datetime, timezone

# Ignore warnings from astropy about ERFA
warnings.filterwarnings("ignore", category=ErfaWarning)


CSV_PATH = "observations_20251116_051444-CCD.csv"
FILTER_START_DATE = True # Whether to filter by start and end date
START_DATE_STR = "2025-01-01" # Adjusted start date for smaller dataset yyyy-mm-dd
END_DATE_STR   = "2026-01-01" # Adjusted end date for smaller dataset yyyy-mm-dd


# Read in the CSV file
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"CSV Columns: {df.columns}")
print(f"Initial rows: {len(df)}")

# Clean and convert data types
df["JD"] = pd.to_numeric(df["jd"], errors="coerce")
df["Magnitude"] = pd.to_numeric(df["mag"], errors="coerce")

# Drop rows with missing JD or Magnitude
df = df.dropna(subset=["JD", "Magnitude"])

# Convert JD to UTC datetime
df["Date"] = Time(df["JD"].values, format="jd").to_datetime()
df["Date"] = pd.to_datetime(df["Date"], utc=True)

# Filter by start and end date if enabled
if FILTER_START_DATE:
    start_dt = pd.to_datetime(START_DATE_STR, utc=True)
    end_dt = pd.to_datetime(END_DATE_STR, utc=True)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

# Only keep magnitudes between 3.4 and 4.4 since these are the values that the star reaches
df = df[(df["Magnitude"] >= 3.4) & (df["Magnitude"] <= 4.4)]
print(f"Cleaned rows: {len(df)}")

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(df["Date"], df["Magnitude"], s=10, color="blue", alpha=0.6)
plt.gca().invert_yaxis()  # Invert y-axis for magnitudes

# Add a vertical line for the current date and time (UTC) and label it
now = datetime.now(timezone.utc)
plt.axvline(x=now, color='gray', linestyle=':', linewidth=1.5, label='Now (UTC)')

# Show legend so the label is visible (adjust location as needed)
plt.legend(loc='upper right', fontsize='small', framealpha=0.8)

# Add labels and title
plt.title("AAVSO CCD V Observations")
plt.xlabel("Date (UTC)")
plt.ylabel("Magnitude (V)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
