# SCRIPT TO DERIVE MAXIMUM WET-BULB TEMPERATURE FROM DOWNSCALED TASMAX AND HURSMIN
# USES THE STULL (2011) EMPIRICAL APPROXIMATION:
#
#   Tw = T * atan(0.151977 * (RH + 8.313659)^0.5)
#        + atan(T + RH)
#        - atan(RH - 1.676331)
#        + 0.00391838 * RH^1.5 * atan(0.023101 * RH)
#        - 4.686035
#
# where T is air temperature in degrees Celsius and RH is relative humidity
# in percent (0-100). Reference: Stull, R. (2011). "Wet-Bulb Temperature from
# Relative Humidity and Air Temperature." J. Appl. Meteor. Climatol., 50,
# 2267-2269.

import xarray as xr
import numpy as np
from dask.distributed import Client

# CONFIGURATION
MODEL_NAME = "MPI-ESM1-2-HR"
BASE_PATH = "gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/daily/"

TASMAX_PATH = f"{BASE_PATH}tasmax/tasmax_{MODEL_NAME}_ww-isimip_ssp585_day.zarr"
HURSMIN_PATH = f"{BASE_PATH}hursmin/hursmin_{MODEL_NAME}_ww-isimip_ssp585_day.zarr"
OUT_PATH = f"{BASE_PATH}wetbulbmax/wetbulbmax_{MODEL_NAME}_ww-isimip_ssp585_day.zarr"


def calculate_wetbulb(tasmax_k, hursmin_pct):
    """
    Calculates maximum wet-bulb temperature from temperature max (Kelvin) and
    minimum relative humidity (percent) using the Stull (2011) formulation.
    """
    # Convert Kelvin to Celsius
    t_c = tasmax_k - 273.15

    # Clip RH to a physically valid range for numerical safety
    rh = hursmin_pct.clip(0, 100)

    # Stull (2011) empirical wet-bulb temperature approximation
    # Standard xarray/numpy math operations automatically handle dask arrays.
    tw = (
        t_c * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )

    return tw.astype("float32")


def generate_wetbulbmax():
    print("Opening downscaled tasmax and hursmin Zarr stores...")
    ds_tasmax = xr.open_zarr(TASMAX_PATH)
    ds_hursmin = xr.open_zarr(HURSMIN_PATH)

    da_tasmax = ds_tasmax["tasmax"]
    da_hursmin = ds_hursmin["hursmin"]

    print("Calculating wetbulbmax...")
    da_wetbulb = calculate_wetbulb(da_tasmax, da_hursmin)
    ds_wetbulb = da_wetbulb.to_dataset(name="wetbulbmax")

    # Ensure coordinates are identical
    ds_wetbulb = ds_wetbulb.assign_coords(ds_tasmax.coords)

    # Clear encoding to prevent chunk mismatch errors during writing
    for var in ds_wetbulb.variables:
        ds_wetbulb[var].encoding.pop("chunks", None)

    print(f"Saving to {OUT_PATH}...")
    ds_wetbulb.to_zarr(OUT_PATH, mode="w")
    print("Done.")


if __name__ == "__main__":
    client = Client()  # starts a local cluster using all available cores
    generate_wetbulbmax()
