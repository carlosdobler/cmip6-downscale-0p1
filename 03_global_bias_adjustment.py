import os
import gc
import time
import logging
import warnings
from datetime import datetime, timedelta
import xarray as xr
import xarray_regrid
import numpy as np
import dask.array as da
import multiprocessing
from ibicus.debias import ISIMIP
from ibicus.variables import tas
from ibicus.utils import get_library_logger

# Silence gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
# Prevent OpenBLAS from over-threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Suppress ibicus logging warnings
ibicus_logger = get_library_logger()
ibicus_logger.setLevel(logging.ERROR)

# Suppress specific ibicus UserWarnings (expected NaNs from masking, and progress bar)
warnings.filterwarnings(
    "ignore", message=".*contains inf or nan values.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*progressbar argument is ignored.*", category=UserWarning
)


# CONFIGURATION
WORKERS = 47
MODEL_NAME = "MPI-ESM1-2-HR"
ERA5_LAND_PATH = (
    "gs://clim_data_reg_useast1/era5_land/daily_aggregates/temperature_2m.zarr"
)
CMIP6_HIST_PATH = "gs://cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/day/tas/gn/v20190710/"
CMIP6_SSP585_PATH = "gs://cmip6/CMIP6/ScenarioMIP/DKRZ/MPI-ESM1-2-HR/ssp585/r1i1p1f1/day/tas/gn/v20190710/"

# OUTPUT PATH
OUTPUT_ZARR_PATH = f"gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/daily/tas/tas_{MODEL_NAME}_ww-isimip_ssp585_day.zarr"

# TIME RANGES
TRAIN_START, TRAIN_END = "1971-01-01", "2010-12-31"
INFER_START, INFER_END = "1961-01-01", "2099-12-31"

# CHUNKING
CHUNK_SIZE = 120


# PREPROCESSING HELPERS
def load_cmip6_aligned(hist_path: str, ssp_path: str) -> xr.Dataset:
    """
    Loads CMIP6 datasets and aligns longitude to -180-180.
    """
    ds_hist = xr.open_zarr(hist_path)
    ds_ssp = xr.open_zarr(ssp_path)
    ds = xr.concat([ds_hist, ds_ssp], dim="time").sel(
        time=slice(INFER_START, INFER_END)
    )
    # Map 0...360 to -180...180
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    return ds.sortby("lon")


def get_cmip6_chunk(
    ds: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.Dataset:
    """
    Slice CMIP6 data to extent, handling circular longitude wrapping.
    No buffer needed for bias adjustment.
    """
    buffer = 2.0  # Small buffer for interpolation safety

    # step 1: slice longitude (circular wrapping)
    lon_min_buf, lon_max_buf = lon_min - buffer, lon_max + buffer

    if lon_min_buf < -180 or lon_max_buf > 180:
        if lon_min_buf < -180:
            west_part = ds.sel(lon=slice(lon_min_buf + 360, 180))
            east_part = ds.sel(lon=slice(-180, lon_max_buf))
            west_part = west_part.assign_coords(lon=west_part.lon - 360)
            ds_buffered = xr.concat([west_part, east_part], dim="lon")
        else:
            west_part = ds.sel(lon=slice(lon_min_buf, 180))
            east_part = ds.sel(lon=slice(-180, lon_max_buf - 360))
            east_part = east_part.assign_coords(lon=east_part.lon + 360)
            ds_buffered = xr.concat([west_part, east_part], dim="lon")
    else:
        ds_buffered = ds.sel(lon=slice(lon_min_buf, lon_max_buf))

    # step 2: slice latitude
    lat_min_buf, lat_max_buf = (
        min(lat_min, lat_max) - buffer,
        max(lat_min, lat_max) + buffer,
    )
    return ds_buffered.sel(lat=slice(lat_min_buf, lat_max_buf)).load()


def initialize_global_zarr():
    print("Initializing global Zarr store...")
    ds_temp = xr.open_zarr(ERA5_LAND_PATH)
    lats, lons = ds_temp.y.values, ds_temp.x.values

    ds_cmip = load_cmip6_aligned(CMIP6_HIST_PATH, CMIP6_SSP585_PATH)
    time_coords = ds_cmip.time

    shape = (len(time_coords), len(lats), len(lons))
    chunks = (1000, CHUNK_SIZE, CHUNK_SIZE)

    ds_out = xr.Dataset(
        data_vars={
            "tas": (
                ("time", "latitude", "longitude"),
                da.full(shape, np.nan, chunks=chunks, dtype=np.float32),
            )
        },
        coords={"time": time_coords, "latitude": lats, "longitude": lons},
    )

    for var in ds_out.variables:
        ds_out[var].encoding.clear()

    for var in ds_out.data_vars:
        ds_out[var].encoding["fill_value"] = np.nan

    ds_out.to_zarr(OUTPUT_ZARR_PATH, compute=False, mode="w", zarr_format=3)
    print("Initialization complete.")


def process_chunk(
    lat_idx_start: int,
    lon_idx_start: int,
    total_count: int,
    current_idx: int,
    land_mask_master: xr.DataArray,
):
    # 1. Determine actual chunk size
    if lat_idx_start == 1680:
        actual_h = 121
    else:
        actual_h = CHUNK_SIZE
    actual_w = CHUNK_SIZE

    lat_idx_end, lon_idx_end = lat_idx_start + actual_h, lon_idx_start + actual_w

    # 2. Land mask skip
    mask_chunk = land_mask_master.isel(
        y=slice(lat_idx_start, lat_idx_end), x=slice(lon_idx_start, lon_idx_end)
    )
    if not mask_chunk.any():
        print(
            f"[{current_idx}/{total_count}] Lat {lat_idx_start}, Lon {lon_idx_start} | 100% ocean: skipping."
        )
        return

    chunk_start_time = time.time()
    time_now = (datetime.now() - timedelta(hours=6)).strftime("%H:%M")
    print(
        f"[{current_idx}/{total_count}] Lat {lat_idx_start}, Lon {lon_idx_start} | Started {time_now}..."
    )

    # 3. Load ERA5-Land (observations)
    ds_obs_full = xr.open_zarr(ERA5_LAND_PATH).temperature_2m
    # Use index-based slicing for ERA5-Land
    obs_chunk = (
        ds_obs_full.isel(
            y=slice(lat_idx_start, lat_idx_end), x=slice(lon_idx_start, lon_idx_end)
        )
        .sel(time=slice(TRAIN_START, TRAIN_END))
        .load()
    )

    # Rename to match standard
    obs_chunk = obs_chunk.rename({"y": "latitude", "x": "longitude"})

    # 4. Load CMIP6
    ds_cmip_full = load_cmip6_aligned(CMIP6_HIST_PATH, CMIP6_SSP585_PATH)
    target_lats = land_mask_master.y.isel(y=slice(lat_idx_start, lat_idx_end)).values
    target_lons = land_mask_master.x.isel(x=slice(lon_idx_start, lon_idx_end)).values

    cmip_chunk_raw = get_cmip6_chunk(
        ds_cmip_full,
        target_lons.min(),
        target_lons.max(),
        target_lats.min(),
        target_lats.max(),
    )

    # 5. Interpolate CMIP6 to 0.1 degree grid
    print("Interpolating CMIP6...")
    target_grid = xr.Dataset(
        coords={
            "latitude": target_lats,
            "longitude": target_lons,
        }
    )
    cmip_chunk_interp = (
        cmip_chunk_raw.rename({"lat": "latitude", "lon": "longitude"})
        .regrid.linear(target_grid)
        .tas.load()
    )

    # 6. Prepare data for ibicus
    # Rename mask dimensions to match our standardized format
    mask_chunk_aligned = mask_chunk.rename({"y": "latitude", "x": "longitude"})

    cm_hist = cmip_chunk_interp.sel(time=slice(TRAIN_START, TRAIN_END)).where(
        mask_chunk_aligned
    )
    cm_fut = cmip_chunk_interp.sel(time=slice(INFER_START, INFER_END)).where(
        mask_chunk_aligned
    )
    obs_hist = obs_chunk.sel(time=slice(TRAIN_START, TRAIN_END)).where(
        mask_chunk_aligned
    )

    # Ibicus expects (time, lat, lon)
    # Ensure dimensions are (time, latitude, longitude) and cast to float32 for speed
    obs_hist_vals = obs_hist.transpose("time", "latitude", "longitude").values.astype(
        np.float32
    )
    cm_hist_vals = cm_hist.transpose("time", "latitude", "longitude").values.astype(
        np.float32
    )
    cm_fut_vals = cm_fut.transpose("time", "latitude", "longitude").values.astype(
        np.float32
    )

    # 7. Apply ibicus ISIMIP debiaser
    print("Applying ISIMIP debiaser...")
    debiaser = ISIMIP.from_variable(tas)

    # We use ibicus's internal parallelization
    res_vals = debiaser.apply(
        obs_hist_vals,
        cm_hist_vals,
        cm_fut_vals,
        time_obs=obs_hist.time.values,
        time_cm_hist=cm_hist.time.values,
        time_cm_future=cm_fut.time.values,
        parallel=True,
        nr_processes=WORKERS,
        failsafe=True,
    )

    # 8. Save result
    ds_res = xr.Dataset(
        data_vars={"tas": (("time", "latitude", "longitude"), res_vals)},
        coords={
            "time": cm_fut.time,
            "latitude": target_lats,
            "longitude": target_lons,
        },
    )

    # remove default formatting (already set in initialized zarr)
    for var in ds_res.variables:
        ds_res[var].encoding = {}

    # drop any variables (like 'height') that don't have the expected dimensions
    # this fixes the ValueError when setting `region` explicitly in to_zarr()
    ds_res = ds_res.drop_vars(
        [
            v
            for v in ds_res.variables
            if not any(
                dim in ds_res[v].dims for dim in ["time", "latitude", "longitude"]
            )
        ]
    )

    ds_res.to_zarr(
        OUTPUT_ZARR_PATH,
        region={
            "time": slice(0, len(cm_fut.time)),
            "latitude": slice(lat_idx_start, lat_idx_end),
            "longitude": slice(lon_idx_start, lon_idx_end),
        },
        zarr_format=3,
    )

    elapsed = (time.time() - chunk_start_time) / 60
    print(f"Done in {elapsed:.2f} min.\n")

    # Cleanup
    del (
        obs_chunk,
        cmip_chunk_raw,
        cmip_chunk_interp,
        cm_hist,
        cm_fut,
        obs_hist,
        res_vals,
        ds_res,
    )
    gc.collect()


def run_global_bias_adjustment():
    multiprocessing.set_start_method("spawn", force=True)

    # Load land mask
    print("Loading land mask...")
    ds_temp = xr.open_zarr(ERA5_LAND_PATH).temperature_2m.isel(time=0).load()
    land_mask = ds_temp != 0.0

    lats_idx = range(0, 1800, CHUNK_SIZE)
    lons_idx = range(0, 3600, CHUNK_SIZE)
    total_chunks = len(lats_idx) * len(lons_idx)

    # # Initialize store if not already done
    # if not os.path.exists(OUTPUT_ZARR_PATH):
    #     initialize_global_zarr()

    print(
        f"Starting global bias adjustment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}..."
    )
    global_start_time = time.time()

    current = 1
    for lat_start_idx in lats_idx:
        for lon_start_idx in lons_idx:
            process_chunk(
                lat_start_idx, lon_start_idx, total_chunks, current, land_mask
            )
            current += 1

    # current = 1
    # for lat_start_idx in reversed(lats_idx):
    #     for lon_start_idx in reversed(lons_idx):
    #         process_chunk(
    #             lat_start_idx, lon_start_idx, total_chunks, current, land_mask
    #         )
    #         current += 1

    total_elapsed = (time.time() - global_start_time) / 3600
    print(f"Global bias adjustment completed in {total_elapsed:.2f} hours.")


if __name__ == "__main__":
    run_global_bias_adjustment()
