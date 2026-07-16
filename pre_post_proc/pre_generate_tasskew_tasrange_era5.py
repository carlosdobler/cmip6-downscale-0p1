import xarray as xr
import numpy as np
import fsspec
from dask.distributed import Client


# CONFIGURATION
ERA5_BASE = "gs://clim_data_reg_useast1/era5_land/daily_aggregates/"
TAS_PATH = f"{ERA5_BASE}temperature_2m.zarr"
TASMAX_PATH = f"{ERA5_BASE}temperature_2m_max.zarr"
TASMIN_PATH = f"{ERA5_BASE}temperature_2m_min.zarr"

# Output paths
RANGE_OUT = f"{ERA5_BASE}temperature_2m_range.zarr"
SKEW_OUT = f"{ERA5_BASE}temperature_2m_skew.zarr"


def generate_derived():
    print("Opening ERA5-Land Zarr stores...")
    ds_tas = xr.open_zarr(TAS_PATH)
    ds_max = xr.open_zarr(TASMAX_PATH)
    ds_min = xr.open_zarr(TASMIN_PATH)

    # Use actual variable names in the datasets
    da_tas = ds_tas["temperature_2m"]
    da_max = ds_max["temperature_2m_max"]
    da_min = ds_min["temperature_2m_min"]

    print("Calculating temperature_2m_range...")
    da_range = (da_max - da_min).astype("float32")
    ds_range = da_range.to_dataset(name="temperature_2m_range")

    # Ensure coordinates are identical to avoid alignment issues later
    ds_range = ds_range.assign_coords(ds_tas.coords)

    # Clear encoding to prevent Zarr from complaining about chunk mismatches
    for var in ds_range.variables:
        ds_range[var].encoding.pop("chunks", None)

    print(f"Saving range to {RANGE_OUT}...")
    ds_range.to_zarr(RANGE_OUT, mode="w", consolidated=False)

    print("Calculating tasskew...")
    # Formula: skew = (mean - min) / (max - min)
    # We clip to [0, 1] to handle floating point noise where mean might be
    # slightly outside min/max bounds due to aggregation methods
    # A runtime warning ("invalid value encountered in divide") will pop up
    # due to ocean cells (zeroes, which leads to range being zero, which
    # is an invalid denominator)
    da_skew = ((da_tas - da_min) / da_range).clip(0, 1).astype("float32")
    ds_skew = da_skew.to_dataset(name="temperature_2m_skew")
    ds_skew = ds_skew.assign_coords(ds_tas.coords)

    for var in ds_skew.variables:
        ds_skew[var].encoding.pop("chunks", None)

    print(f"Saving skew to {SKEW_OUT}...")
    ds_skew.to_zarr(SKEW_OUT, mode="w", consolidated=False)
    print("Done.")


if __name__ == "__main__":
    client = Client()  # starts a local cluster using all available cores
    generate_derived()
