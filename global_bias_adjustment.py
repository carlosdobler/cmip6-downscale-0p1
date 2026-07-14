import argparse
import os
import gc
import time
import logging
import warnings
from datetime import datetime, timedelta
import xarray as xr
import xarray_regrid
import fsspec
import numpy as np
import dask.array as da
import multiprocessing
import yaml
from ibicus.debias import ISIMIP
from ibicus.variables import tas, pr, tasrange, tasskew, hurs, rsds, sfcwind
from ibicus.utils import get_library_logger

# Prevent OpenBLAS from over-threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Suppress ibicus logging warnings
ibicus_logger = get_library_logger()
ibicus_logger.setLevel(logging.CRITICAL)

# Suppress specific ibicus UserWarnings (expected NaNs from masking, and progress bar)
warnings.filterwarnings(
    "ignore", message=".*contains inf or nan values.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*progressbar argument is ignored.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in divide", category=RuntimeWarning
)


# CONFIGURATION
WORKERS = 47
MODEL_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.yaml")


def load_model_registry(path: str = MODEL_REGISTRY_PATH) -> dict:
    """
    Loads the model registry YAML file mapping model aliases (lowercased
    full CMIP6 model names) to their GCS path metadata.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


VARIABLE_CONFIG = {
    "tas": {
        "ibicus_var": tas,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/temperature_2m.zarr",
        "era5_var": "temperature_2m",
        "cmip6_var": "tas",
        "interpolation": "linear",
    },
    "pr": {
        "ibicus_var": pr,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/total_precipitation_sum.zarr",
        "era5_var": "total_precipitation_sum",
        "cmip6_var": "pr",
        "interpolation": "conservative",
    },
    "tasrange": {
        "ibicus_var": tasrange,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/temperature_2m_range.zarr",
        "era5_var": "temperature_2m_range",
        "cmip6_var": "tasrange",
        "interpolation": "linear",
    },
    "tasskew": {
        "ibicus_var": tasskew,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/temperature_2m_skew.zarr",
        "era5_var": "temperature_2m_skew",
        "cmip6_var": "tasskew",
        "interpolation": "linear",
    },
    "hurs": {
        "ibicus_var": hurs,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/relative_humidity.zarr",
        "era5_var": "relative_humidity",
        "cmip6_var": "hurs",
        "interpolation": "linear",
    },
    "hursmin": {
        "ibicus_var": hurs,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/relative_humidity_min.zarr",
        "era5_var": "relative_humidity_min",
        "cmip6_var": "hursmin",
        "interpolation": "linear",
    },
    "rsds": {
        "ibicus_var": rsds,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/surface_solar_radiation_downwards_sum.zarr",
        "era5_var": "surface_solar_radiation_downwards_sum",
        "cmip6_var": "rsds",
        "interpolation": "conservative",
    },
    "sfcwind": {
        "ibicus_var": sfcwind,
        "era5_path": "gs://clim_data_reg_useast1/era5_land/daily_aggregates/wind_speed_10m.zarr",
        "era5_var": "wind_speed",
        "cmip6_var": "sfcWind",
        "interpolation": "linear",
    },
}


# The following variables will be set dynamically in run_global_bias_adjustment
VARIABLE = None
VAR_SETTINGS = None
ERA5_LAND_PATH = None
OUTPUT_ZARR_PATH = None
STATUS_DIR = None
MODEL_CONFIG = None
MODEL_NAME = None

# TIME RANGES
TRAIN_START, TRAIN_END = "1971-01-01", "2010-12-31"
INFER_START, INFER_END = "1961-01-01", "2099-12-31"

# CHUNKING
CHUNK_SIZE = 120


# PREPROCESSING HELPERS
def get_cmip6_path(var_name: str, experiment: str) -> str:
    """
    Returns the GCS path for a CMIP6 variable and experiment.
    """
    if experiment == "historical":
        activity = "CMIP"
        institution = MODEL_CONFIG["institution_hist"]
        version = MODEL_CONFIG["version_hist"]
    else:
        activity = "ScenarioMIP"
        institution = MODEL_CONFIG["institution_ssp"]
        version = MODEL_CONFIG["version_ssp"]

    return (
        f"gs://cmip6/CMIP6/{activity}/{institution}/{MODEL_NAME}/{experiment}/"
        f"{MODEL_CONFIG['ensemble_member']}/day/{var_name}/"
        f"{MODEL_CONFIG['grid_label']}/{version}/"
    )


def load_cmip6_simple(var_name: str) -> xr.Dataset:
    """
    Loads CMIP6 datasets and aligns longitude to -180-180.
    """
    hist_path = get_cmip6_path(var_name, "historical")
    ssp_path = get_cmip6_path(var_name, "ssp585")

    ds_hist = xr.open_zarr(hist_path)
    ds_ssp = xr.open_zarr(ssp_path)
    ds = xr.concat([ds_hist, ds_ssp], dim="time").sel(
        time=slice(INFER_START, INFER_END)
    )
    # Map 0...360 to -180...180
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    return ds.sortby("lon")


def load_cmip6_aligned(var_name: str) -> xr.Dataset:
    """
    Loads CMIP6 datasets and handles derived variables (Choice B).
    """
    if var_name == "tasrange":
        print("Calculating CMIP6 tasrange on-the-fly...")
        ds_max = load_cmip6_simple("tasmax")
        ds_min = load_cmip6_simple("tasmin")
        # Alignment is handled in load_cmip6_simple, so we can subtract
        da = ds_max["tasmax"] - ds_min["tasmin"]
        return da.to_dataset(name="tasrange")

    elif var_name == "tasskew":
        print("Calculating CMIP6 tasskew on-the-fly...")
        ds_max = load_cmip6_simple("tasmax")
        ds_min = load_cmip6_simple("tasmin")
        ds_tas = load_cmip6_simple("tas")
        # Formula: (mean - min) / (max - min)
        # Use .clip(0, 1) to ensure physical validity despite numerical noise
        da = (ds_tas["tas"] - ds_min["tasmin"]) / (ds_max["tasmax"] - ds_min["tasmin"])
        return da.clip(0, 1).to_dataset(name="tasskew")

    elif var_name == "hursmin":
        # tasmin is used here as a substitute of dewpoint/min_dewpoint. Daily tasmin is readily
        # available in CMIP6.
        # Daily tasmin serves as a meteorological proxy for both average and minimum dewpoint. This
        # substitution relies on the principle that nighttime cooling frequently causes the lower
        # atmosphere to reach saturation, bringing the air temperature down to equal the dewpoint.
        # Since the absolute water vapor content of an air mass typically remains conservative
        # throughout a 24-hour cycle, this overnight baseline adequately represents the daytime
        # moisture levels as well (i.e., dewpoint is relatively constant throughout the day). By
        # plugging tasmin into the numerator of the relative humidity equation, we can reliably
        # estimate the daily actual vapor pressure without needing a specifically measured minimum
        # dewpoint.
        # Note: this approach may overestimate humidity in arid climates where nighttime saturation
        # is rare.

        print("Calculating CMIP6 hursmin on-the-fly...")
        ds_max = load_cmip6_simple("tasmax")
        ds_min = load_cmip6_simple("tasmin")

        t_c = ds_max["tasmax"] - 273.15
        td_c = ds_min["tasmin"] - 273.15

        a = 17.625
        b = 243.04

        # RH = 100 * exp((a * Td) / (b + Td) - (a * T) / (b + T))
        rh = 100 * np.exp((a * td_c) / (b + td_c) - (a * t_c) / (b + t_c))

        # Clip RH to [0, 100] as relative humidity cannot be negative or exceed 100%
        return rh.clip(0, 100).to_dataset(name="hursmin")

    else:
        return load_cmip6_simple(var_name)


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
    # Increase buffer to 4.0 for sequential interpolation safety
    buffer = 4.0

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


def interpolate_cmip6(
    ds: xr.Dataset, target_grid: xr.Dataset, method: str
) -> xr.DataArray:
    """
    Interpolates CMIP6 to target grid using specified method.
    """
    ds_renamed = ds.rename({"lat": "latitude", "lon": "longitude"})

    if method == "linear":
        return ds_renamed.regrid.linear(target_grid)[VAR_SETTINGS["cmip6_var"]]
    elif method == "conservative":
        # Sequential interpolation: 1.6 -> 0.8 -> 0.4 -> 0.2 -> 0.1
        resolutions = [1.6, 0.8, 0.4, 0.2]
        current_ds = ds_renamed

        target_lats = target_grid.latitude.values
        target_lons = target_grid.longitude.values
        lat_min, lat_max = target_lats.min(), target_lats.max()
        lon_min, lon_max = target_lons.min(), target_lons.max()

        for res in resolutions:
            # Create intermediate grid aligned with ERA5-Land (multiples of 0.1)
            # We anchor to 90N and -180E to ensure alignment since res is a multiple of 0.1
            grid_lat_max = 90.0 - np.floor((90.0 - lat_max) / res) * res
            grid_lat_min = 90.0 - np.ceil((90.0 - lat_min) / res) * res
            grid_lon_min = -180.0 + np.floor((lon_min - (-180.0)) / res) * res
            grid_lon_max = -180.0 + np.ceil((lon_max - (-180.0)) / res) * res

            # Ensure we don't exceed global bounds
            grid_lat_max = min(grid_lat_max, 90.0)
            grid_lat_min = max(grid_lat_min, -90.0)

            inter_lats = np.arange(grid_lat_max, grid_lat_min - res / 10, -res)
            inter_lons = np.arange(grid_lon_min, grid_lon_max + res / 10, res)

            inter_grid = xr.Dataset(
                coords={
                    "latitude": inter_lats,
                    "longitude": inter_lons,
                }
            )
            current_ds = current_ds.regrid.conservative(inter_grid)

        # Final step to 0.1 (target_grid)
        return current_ds.regrid.conservative(target_grid)[VAR_SETTINGS["cmip6_var"]]
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def initialize_global_zarr():
    print("Initializing global Zarr store...")
    ds_temp = xr.open_zarr(ERA5_LAND_PATH)
    lats, lons = ds_temp.y.values, ds_temp.x.values

    ds_cmip = load_cmip6_aligned(VAR_SETTINGS["cmip6_var"])
    time_coords = ds_cmip.time

    shape = (len(time_coords), len(lats), len(lons))
    chunks = (1000, CHUNK_SIZE, CHUNK_SIZE)

    ds_out = xr.Dataset(
        data_vars={
            VARIABLE: (
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


def get_chunk_id(lat_idx: int, lon_idx: int) -> str:
    return f"lat_{lat_idx}_lon_{lon_idx}"


def mark_chunk_done(chunk_id: str):
    fs = fsspec.filesystem("gs")
    done_file = f"{STATUS_DIR}{chunk_id}.done"
    started_file = f"{STATUS_DIR}{chunk_id}.started"
    fs.touch(done_file)
    if fs.exists(started_file):
        fs.rm(started_file)


def process_chunk(
    lat_idx_start: int,
    lon_idx_start: int,
    total_count: int,
    current_idx: int,
    land_mask_master: xr.DataArray,
    ds_obs_full: xr.DataArray,
    ds_cmip_full: xr.Dataset,
):
    chunk_id = get_chunk_id(lat_idx_start, lon_idx_start)

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
        mark_chunk_done(chunk_id)
        return

    chunk_start_time = time.time()
    time_now = (datetime.now() - timedelta(hours=6)).strftime("%H:%M")
    print(
        f"[{current_idx}/{total_count}] Lat {lat_idx_start}, Lon {lon_idx_start} | Started {time_now}..."
    )

    # 3. Load ERA5-Land (observations)
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
    target_lats = land_mask_master.y.isel(y=slice(lat_idx_start, lat_idx_end)).values
    target_lons = land_mask_master.x.isel(x=slice(lon_idx_start, lon_idx_end)).values

    cmip_chunk_raw = get_cmip6_chunk(
        ds_cmip_full,
        target_lons.min(),
        target_lons.max(),
        target_lats.min(),
        target_lats.max(),
    )

    # Drop the bnds dimensions/variables if they exist, as they break xarray_regrid conservative method
    if "bnds" in cmip_chunk_raw.dims:
        cmip_chunk_raw = cmip_chunk_raw.drop_dims("bnds")
    if "lat_bnds" in cmip_chunk_raw.variables:
        cmip_chunk_raw = cmip_chunk_raw.drop_vars(["lat_bnds"])
    if "lon_bnds" in cmip_chunk_raw.variables:
        cmip_chunk_raw = cmip_chunk_raw.drop_vars(["lon_bnds"])
    if "time_bnds" in cmip_chunk_raw.variables:
        cmip_chunk_raw = cmip_chunk_raw.drop_vars(["time_bnds"])

    # 5. Interpolate CMIP6 to 0.1 degree grid
    print("Interpolating CMIP6...")
    target_grid = xr.Dataset(
        coords={
            "latitude": target_lats,
            "longitude": target_lons,
        }
    )
    cmip_chunk_interp = interpolate_cmip6(
        cmip_chunk_raw, target_grid, method=VAR_SETTINGS["interpolation"]
    ).load()

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

    if VARIABLE == "pr":
        # Convert ERA5-Land precipitation from m/day to kg m-2 s-1
        obs_hist_vals = obs_hist_vals * (1000.0 / 86400.0)
        # Clip negative precipitation values (often ~ -1e-8 due to numerical inaccuracies)
        obs_hist_vals = np.clip(obs_hist_vals, 0, None)
        cm_hist_vals = np.clip(cm_hist_vals, 0, None)
        cm_fut_vals = np.clip(cm_fut_vals, 0, None)

    if VARIABLE in ["hurs", "hursmin"]:
        # Clip to ibicus reasonable physical range for hurs [1e-5, 150]
        obs_hist_vals = np.clip(obs_hist_vals, 1e-5, 150)
        cm_hist_vals = np.clip(cm_hist_vals, 1e-5, 150)
        cm_fut_vals = np.clip(cm_fut_vals, 1e-5, 150)

    if VARIABLE == "sfcwind":
        # Clip to ibicus reasonable physical range for sfcwind [1e-5, 500]
        # This prevents warnings when wind speed is 0.0
        obs_hist_vals = np.clip(obs_hist_vals, 1e-5, 500)
        cm_hist_vals = np.clip(cm_hist_vals, 1e-5, 500)
        cm_fut_vals = np.clip(cm_fut_vals, 1e-5, 500)

    if VARIABLE == "rsds":
        # Convert ERA5-Land radiation sum (J m-2) to daily mean flux (W m-2)
        # 1 day = 86400 seconds
        obs_hist_vals = obs_hist_vals / 86400.0
        # Clip to ibicus reasonable physical range for rsds [0, 1000]
        obs_hist_vals = np.clip(obs_hist_vals, 0, 1000)
        cm_hist_vals = np.clip(cm_hist_vals, 0, 1000)
        cm_fut_vals = np.clip(cm_fut_vals, 0, 1000)

    # 7. Apply ibicus ISIMIP debiaser
    print("Applying ISIMIP debiaser...")
    debiaser = ISIMIP.from_variable(
        VAR_SETTINGS["ibicus_var"], running_window_step_length=3
    )

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
        data_vars={VARIABLE: (("time", "latitude", "longitude"), res_vals)},
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

    mark_chunk_done(chunk_id)

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


def is_everything_done(expected_count: int) -> bool:
    """
    Checks if all chunks are marked as .done in STATUS_DIR.
    """
    fs = fsspec.filesystem("gs")
    # Using glob to get all .done files
    done_files = fs.glob(f"{STATUS_DIR}*.done")
    # Filter to count only actual data chunks (names starting with 'lat_')
    chunk_done_count = sum(
        1 for f in done_files if os.path.basename(f).startswith("lat_")
    )
    return chunk_done_count >= expected_count


def finalize_longitude_wrap(total_chunks: int):
    """
    Copies data from longitude index 0 (-180.0) to index 3600 (180.0).
    Executed once all chunks are processed.
    """
    fs = fsspec.filesystem("gs")
    wrap_done_file = f"{STATUS_DIR}longitude_wrap.done"
    wrap_started_file = f"{STATUS_DIR}longitude_wrap.started"

    if fs.exists(wrap_done_file):
        print("Longitude wrap already finalized.")
        return

    if not is_everything_done(total_chunks):
        print("Not all chunks are done yet. Skipping longitude wrap for now.")
        return

    # Atomic lock: create .started file
    if fs.exists(wrap_started_file):
        print("Longitude wrap is being processed by another VM.")
        return

    try:
        fs.touch(wrap_started_file)
    except Exception as e:
        print(f"Failed to lock longitude wrap: {e}")
        return

    print("Finalizing longitude wrap (copying index 0 to 3600)...")
    ds_out = xr.open_zarr(OUTPUT_ZARR_PATH)

    # Process in latitude chunks to stay memory-efficient
    lats_idx = range(0, 1800, CHUNK_SIZE)

    for lat_start_idx in lats_idx:
        if lat_start_idx == 1680:
            actual_h = 121
        else:
            actual_h = CHUNK_SIZE
        lat_idx_end = lat_start_idx + actual_h

        # Load column 0 data
        col0_data = (
            ds_out[VARIABLE]
            .isel(latitude=slice(lat_start_idx, lat_idx_end), longitude=0)
            .load()
        )

        # Create dataset for the 180.0 longitude (index 3600)
        ds_wrap = xr.Dataset(
            data_vars={
                VARIABLE: (
                    ("time", "latitude", "longitude"),
                    col0_data.values[:, :, np.newaxis],
                )
            },
            coords={
                "time": col0_data.time,
                "latitude": col0_data.latitude,
                "longitude": [ds_out.longitude.values[3600]],
            },
        )

        # Clear encoding and drop unrelated variables
        for var in ds_wrap.variables:
            ds_wrap[var].encoding = {}
        ds_wrap = ds_wrap.drop_vars(
            [
                v
                for v in ds_wrap.variables
                if not any(
                    dim in ds_wrap[v].dims for dim in ["time", "latitude", "longitude"]
                )
            ]
        )

        # Save to index 3600
        ds_wrap.to_zarr(
            OUTPUT_ZARR_PATH,
            region={
                "time": slice(0, len(col0_data.time)),
                "latitude": slice(lat_start_idx, lat_idx_end),
                "longitude": slice(3600, 3601),
            },
            zarr_format=3,
        )
        print(f"Wrapped latitude slice {lat_start_idx}:{lat_idx_end}")

    fs.touch(wrap_done_file)
    if fs.exists(wrap_started_file):
        fs.rm(wrap_started_file)
    print("Longitude wrap complete.")


def finalize_south_pole(total_chunks: int):
    """
    Copies data from latitude index 1799 (-89.9) to index 1800 (-90.0)
    ONLY if index 1800 is not fully populated (compared to index 1799).
    """
    fs = fsspec.filesystem("gs")
    pole_done_file = f"{STATUS_DIR}south_pole.done"
    pole_started_file = f"{STATUS_DIR}south_pole.started"

    if fs.exists(pole_done_file):
        print("South pole already finalized.")
        return

    if not is_everything_done(total_chunks):
        print("Not all chunks are done yet. Skipping South Pole for now.")
        return

    # Atomic lock: create .started file
    if fs.exists(pole_started_file):
        print("South Pole is being processed by another VM.")
        return

    try:
        fs.touch(pole_started_file)
    except Exception as e:
        print(f"Failed to lock South Pole: {e}")
        return

    print("Checking if South Pole row (index 1800) needs filling...")
    ds_out = xr.open_zarr(OUTPUT_ZARR_PATH)

    # Check if index 1800 is fully populated by comparing valid pixel counts with row 1799
    # Sample first time step for speed
    expected_valid_pixels = int(
        ds_out[VARIABLE].isel(latitude=1799, time=0).notnull().sum().values
    )
    current_valid_pixels = int(
        ds_out[VARIABLE].isel(latitude=1800, time=0).notnull().sum().values
    )

    if current_valid_pixels >= expected_valid_pixels and expected_valid_pixels > 0:
        print("South Pole is fully populated. Skipping fill.")
        fs.touch(pole_done_file)
        if fs.exists(pole_started_file):
            fs.rm(pole_started_file)
        return

    print(
        f"South Pole incomplete (Expected: {expected_valid_pixels}, Found: {current_valid_pixels}). Copying index 1799 to 1800..."
    )

    # Process in longitude chunks to stay memory-efficient
    lons_idx = range(0, 3601, CHUNK_SIZE)

    for lon_start_idx in lons_idx:
        lon_idx_end = min(lon_start_idx + CHUNK_SIZE, 3601)

        # Load row 1799 data
        row1799_data = (
            ds_out[VARIABLE]
            .isel(latitude=1799, longitude=slice(lon_start_idx, lon_idx_end))
            .load()
        )

        # Create dataset for the -90.0 latitude (index 1800)
        ds_pole = xr.Dataset(
            data_vars={
                VARIABLE: (
                    ("time", "latitude", "longitude"),
                    row1799_data.values[:, np.newaxis, :],
                )
            },
            coords={
                "time": row1799_data.time,
                "latitude": [ds_out.latitude.values[1800]],
                "longitude": row1799_data.longitude,
            },
        )

        # Clear encoding and drop unrelated variables
        for var in ds_pole.variables:
            ds_pole[var].encoding = {}
        ds_pole = ds_pole.drop_vars(
            [
                v
                for v in ds_pole.variables
                if not any(
                    dim in ds_pole[v].dims for dim in ["time", "latitude", "longitude"]
                )
            ]
        )

        # Save to index 1800
        ds_pole.to_zarr(
            OUTPUT_ZARR_PATH,
            region={
                "time": slice(0, len(row1799_data.time)),
                "latitude": slice(1800, 1801),
                "longitude": slice(lon_start_idx, lon_idx_end),
            },
            zarr_format=3,
        )
        print(f"Filled South Pole for longitude slice {lon_start_idx}:{lon_idx_end}")

    fs.touch(pole_done_file)
    if fs.exists(pole_started_file):
        fs.rm(pole_started_file)
    print("South pole finalization complete.")


def run_global_bias_adjustment():
    multiprocessing.set_start_method("spawn", force=True)

    # Load land mask
    print("Loading land mask...")

    # ALWAYS load the land mask from the reference 'tas' (temperature_2m) variable.
    # This prevents issues where derived variables (like 'tasskew') use np.nan instead of 0.0
    # as the fill value over the ocean, which would break the mask generation (NaN != 0.0 is True).
    tas_path = VARIABLE_CONFIG["tas"]["era5_path"]
    tas_var = VARIABLE_CONFIG["tas"]["era5_var"]
    ds_temp = xr.open_zarr(tas_path)[tas_var].isel(time=0).load()

    # Ensure it's not exactly 0.0 AND not NaN
    land_mask = (ds_temp != 0.0) & ds_temp.notnull()

    lats_idx = range(0, 1800, CHUNK_SIZE)
    lons_idx = range(0, 3600, CHUNK_SIZE)
    total_chunks = len(lats_idx) * len(lons_idx)

    # Initialize store if not already done
    fs = fsspec.filesystem("gs")
    # Check for Zarr V3 metadata file to ensure it exists
    if not fs.exists(f"{OUTPUT_ZARR_PATH}/zarr.json"):
        initialize_global_zarr()

    # Ensure status directory exists
    if not fs.exists(STATUS_DIR):
        fs.makedirs(STATUS_DIR)

    print(
        f"Starting global bias adjustment for {VARIABLE} at {(datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')}..."
    )
    global_start_time = time.time()

    print("Loading global datasets once to prevent GC/fsspec loop issues...")
    ds_obs_full = xr.open_zarr(ERA5_LAND_PATH)[VAR_SETTINGS["era5_var"]]
    ds_cmip_full = load_cmip6_aligned(VAR_SETTINGS["cmip6_var"])

    current = 1
    for lat_start_idx in lats_idx:
        for lon_start_idx in lons_idx:
            chunk_id = get_chunk_id(lat_start_idx, lon_start_idx)
            done_file = f"{STATUS_DIR}{chunk_id}.done"
            started_file = f"{STATUS_DIR}{chunk_id}.started"

            # Check if chunk is already processed or being processed
            if fs.exists(done_file) or fs.exists(started_file):
                # print(f"[{current}/{total_chunks}] Skipping {chunk_id} (already done or started).")
                current += 1
                continue

            # Atomic lock: create .started file
            try:
                fs.touch(started_file)
            except Exception as e:
                print(f"Failed to lock {chunk_id}: {e}")
                current += 1
                continue

            process_chunk(
                lat_start_idx,
                lon_start_idx,
                total_chunks,
                current,
                land_mask,
                ds_obs_full,
                ds_cmip_full,
            )
            current += 1

    # Copy longitude index 0 to index 3600 to complete the global wrap
    finalize_longitude_wrap(total_chunks)

    # Copy latitude index 1799 to index 1800 to fill missing South Pole
    finalize_south_pole(total_chunks)

    total_elapsed = (time.time() - global_start_time) / 3600
    print(f"Global bias adjustment completed in {total_elapsed:.2f} hours.")

    # Clear fsspec instance cache to prevent noise/RuntimeErrors during interpreter shutdown
    fsspec.AbstractFileSystem.clear_instance_cache()


if __name__ == "__main__":
    model_registry = load_model_registry()

    parser = argparse.ArgumentParser(description="Run global bias adjustment.")
    parser.add_argument(
        "-v",
        "--variable",
        type=str,
        default="tas",
        choices=[
            "tas",
            "pr",
            "tasrange",
            "tasskew",
            "hurs",
            "hursmin",
            "rsds",
            "sfcwind",
        ],
        help="Variable to downscale (tas, pr, tasrange, tasskew, hurs, hursmin, rsds, or sfcwind).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=list(model_registry.keys()),
        help=(
            "CMIP6 model to downscale, as a key in model_registry.yaml "
            "(the full model name, lowercased). Use add_cmip6_model.py to "
            "register a new model."
        ),
    )
    args = parser.parse_args()

    # Set dynamic variables globally
    VARIABLE = args.variable
    VAR_SETTINGS = VARIABLE_CONFIG[VARIABLE]
    ERA5_LAND_PATH = VAR_SETTINGS["era5_path"]

    MODEL_CONFIG = model_registry[args.model]
    MODEL_NAME = MODEL_CONFIG["model_name"]

    # OUTPUT_ZARR_PATH is based on the VARIABLE (tas, pr, tasrange, or tasskew)
    OUTPUT_ZARR_PATH = f"gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/daily/{VARIABLE}/{VARIABLE}_{MODEL_NAME}_ww-isimip_ssp585_day.zarr"
    STATUS_DIR = f"gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/status/{VARIABLE}/{MODEL_NAME}/"

    run_global_bias_adjustment()
