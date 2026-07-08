# SCRIPT TO DERIVE MINIMUM RELATIVE HUMIDITY FROM ERA5-LAND TEMPERATURE MAX AND DEWPOINT MIN
# USING THE AUGUST-ROCHE-MAGNUS EQUATION

# Relative humidity is a ratio comparing the actual vapor pressure in the air to the saturation
# vapor pressure. To calculate the lowest possible relative humidity (hursmin) for a given day,
# we must minimize this fraction. Using the daily maximum temperature (tasmax) maximizes the
# denominator, as hot air has the highest possible capacity to hold water vapor. Pairing it with
# the minimum dewpoint (dmin) minimizes the numerator by representing the lowest absolute
# moisture content present that day. Meteorologically, these extremes typically align in the late
# afternoon when peak surface heating drives turbulent mixing that pulls drier air down from aloft.
# Consequently, combining tasmax and dmin accurately captures the specific atmospheric conditions
# when the air is simultaneously at its hottest and driest.

import xarray as xr
import numpy as np
import warnings
from dask.distributed import Client

# Suppress Zarr V3 consolidation warnings
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
)

# CONFIGURATION
ERA5_BASE = "gs://clim_data_reg_useast1/era5_land/daily_aggregates/"
TAS_PATH = f"{ERA5_BASE}temperature_2m_max.zarr"
D2M_PATH = f"{ERA5_BASE}dewpoint_temperature_2m_min.zarr"
OUT_PATH = f"{ERA5_BASE}relative_humidity_min.zarr"


def calculate_hursmin(tasmax_k, d2mmin_k):
    """
    Calculates minimum relative humidity from temperature max and dewpoint min using the August-Roche-Magnus equation.
    """
    # Convert Kelvin to Celsius
    t_c = tasmax_k - 273.15
    td_c = d2mmin_k - 273.15

    # Constants for August-Roche-Magnus
    a = 17.625
    b = 243.04

    # RH_min = 100 * exp((a * Td_min) / (b + Td_min) - (a * T_max) / (b + T_max))
    # Standard xarray math operations automatically handle dask arrays.
    rh = 100 * np.exp((a * td_c) / (b + td_c) - (a * t_c) / (b + t_c))

    return rh.astype("float32")


def generate_derived():
    print("Opening ERA5-Land Zarr stores...")
    # Open datasets natively as Zarr V3
    ds_tas = xr.open_zarr(TAS_PATH, zarr_format=3)
    ds_d2m = xr.open_zarr(D2M_PATH, zarr_format=3)

    da_tas = ds_tas["temperature_2m_max"]
    da_d2m = ds_d2m["dewpoint_temperature_2m_min"]

    print("Calculating relative_humidity_min...")
    da_rh = calculate_hursmin(da_tas, da_d2m)
    ds_rh = da_rh.to_dataset(name="relative_humidity_min")

    # Ensure coordinates are identical
    ds_rh = ds_rh.assign_coords(ds_tas.coords)

    # Clear encoding to prevent chunk mismatch errors during writing
    for var in ds_rh.variables:
        ds_rh[var].encoding.pop("chunks", None)

    print(f"Saving to {OUT_PATH}...")
    ds_rh.to_zarr(OUT_PATH, mode="w", zarr_format=3)
    print("Done.")


if __name__ == "__main__":
    client = Client()  # starts a local cluster using all available cores
    generate_derived()
