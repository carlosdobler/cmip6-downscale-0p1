# SCRIPT TO DERIVE RELATIVE HUMIDITY FROM ERA5-LAND TEMPERATURE AND DEWPOINT
# USES THE AUGUST-ROCHE-MAGNUS EQUATION

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
TAS_PATH = f"{ERA5_BASE}temperature_2m.zarr"
D2M_PATH = f"{ERA5_BASE}dewpoint_temperature_2m.zarr"
OUT_PATH = f"{ERA5_BASE}relative_humidity.zarr"


def calculate_hurs(tas_k, d2m_k):
    """
    Calculates relative humidity from temperature and dewpoint using the August-Roche-Magnus equation.
    """
    # Convert Kelvin to Celsius
    t_c = tas_k - 273.15
    td_c = d2m_k - 273.15

    # Constants for August-Roche-Magnus
    a = 17.625
    b = 243.04

    # RH = 100 * exp((a * Td) / (b + Td) - (a * T) / (b + T))
    # Standard xarray math operations automatically handle dask arrays.
    rh = 100 * np.exp((a * td_c) / (b + td_c) - (a * t_c) / (b + t_c))

    return rh.astype("float32")


def generate_derived():
    print("Opening ERA5-Land Zarr stores...")
    # Open datasets natively as Zarr V3
    ds_tas = xr.open_zarr(TAS_PATH, zarr_format=3)
    ds_d2m = xr.open_zarr(D2M_PATH, zarr_format=3)

    da_tas = ds_tas["temperature_2m"]
    da_d2m = ds_d2m["dewpoint_temperature_2m"]

    print("Calculating relative_humidity...")
    da_rh = calculate_hurs(da_tas, da_d2m)
    ds_rh = da_rh.to_dataset(name="relative_humidity")

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
