# SCRIPT TO DERIVE WIND SPEED FROM ERA5-LAND U AND V COMPONENTS
# USES THE PYTHAGOREAN THEOREM: wind_speed = sqrt(u^2 + v^2)

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
U10_PATH = f"{ERA5_BASE}u_component_of_wind_10m.zarr"
V10_PATH = f"{ERA5_BASE}v_component_of_wind_10m.zarr"
OUT_PATH = f"{ERA5_BASE}wind_speed_10m.zarr"


def calculate_wind_speed(u, v):
    """
    Calculates wind speed from u and v components.
    """
    # Standard xarray math operations automatically handle dask arrays.
    ws = np.sqrt(u**2 + v**2)

    return ws.astype("float32")


def generate_derived():
    print("Opening ERA5-Land Zarr stores...")
    # Open datasets natively as Zarr V3
    ds_u = xr.open_zarr(U10_PATH, zarr_format=3)
    ds_v = xr.open_zarr(V10_PATH, zarr_format=3)

    da_u = ds_u["u_component_of_wind_10m"]
    da_v = ds_v["v_component_of_wind_10m"]

    print("Calculating wind_speed...")
    da_ws = calculate_wind_speed(da_u, da_v)
    ds_ws = da_ws.to_dataset(name="wind_speed")

    # Ensure coordinates are identical
    ds_ws = ds_ws.assign_coords(ds_u.coords)

    # Clear encoding to prevent chunk mismatch errors during writing
    for var in ds_ws.variables:
        ds_ws[var].encoding.pop("chunks", None)

    print(f"Saving to {OUT_PATH}...")
    ds_ws.to_zarr(OUT_PATH, mode="w", zarr_format=3)
    print("Done.")


if __name__ == "__main__":
    client = Client()  # starts a local cluster using all available cores
    generate_derived()
