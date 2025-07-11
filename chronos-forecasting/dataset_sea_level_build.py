import pandas as pd
import numpy as np

# Load the data, skipping the HDR header (look for line starting after 'Header_End')
with open("C:/Users/picca/Documents/Umberto/Deep_natural_language_proc/project/dataset/sea_level/Sea_level_data_NASA.txt") as f:
    lines = f.readlines()

# Find the first non-header line (assumes header ends with 'Header_End')
data_start = next(i for i, line in enumerate(lines) if "Header_End" in line) + 1

# Read data into DataFrame
data = pd.read_csv("C:/Users/picca/Documents/Umberto/Deep_natural_language_proc/project/dataset/sea_level/Sea_level_data_NASA.txt", 
                   sep='\s+', 
                   header=None, 
                   skiprows=data_start,
                   names=[
                       "altimeter_type", "cycle", "year_fraction", "n_obs", "n_weighted_obs",
                       "gmsl_raw", "gmsl_raw_std", "gmsl_raw_smooth",
                       "gmsl_gia", "gmsl_gia_std", "gmsl_gia_smooth",
                       "gmsl_gia_smooth_detrended", "gmsl_raw_smooth_detrended"
                   ])

# Replace bad values
data.replace(99900.000, np.nan, inplace=True)

# Sort by date
data.sort_values("year_fraction", inplace=True)

# Forward-fill missing values method='ffill', inplace=True
data.ffill(inplace=True)

# Estrarre anno intero e frazione
data["year"] = data["year_fraction"].astype(int)
data["fraction"] = data["year_fraction"] - data["year"]

# Calcolare la data
data["date"] = pd.to_datetime(data["year"].astype(str)) + pd.to_timedelta(data["fraction"] * 365.25, unit="D")

# Impostare come indice se vuoi lavorare come time series
data.set_index("date", inplace=True)

# Extract time series
columns_ts = ["gmsl_raw", "gmsl_raw_smooth",
                       "gmsl_gia", "gmsl_gia_smooth",
                       "gmsl_gia_smooth_detrended", "gmsl_raw_smooth_detrended"]

# Quick visualization (optional)
import matplotlib.pyplot as plt
plt.ylabel("mm")
plt.xlabel("Year")
plt.grid()
for col in columns_ts:
    ts = data[col]
    ts.plot(title="Global Mean Sea Level")
plt.show()

ts_best = data["gmsl_gia_smooth_detrended"]
plt.ylabel("mm")
plt.xlabel("Year")
plt.grid()
ts_best.plot(title="(GIA applied, smoothed, de-trended)")
plt.show()