# SCRIPT TO DOWNLOAD ERA5-LAND DAILY DATA
# RUNNING THIS SCRIPT REQUIRES ~60 GB OF MEMORY
#
# Time chunks are set to OUT_T_CHUNK = 500 days on first write (~57.6 MB at 120×120 spatial).
# Spatial chunks of 120×120 align with global_bias_adjustment.py's CHUNK_SIZE = 120.
# Subsequent yearly appends use safe_chunks=False because each year's data crosses the
# 500-day chunk boundary, requiring a read-modify-write on the last partial chunk. This is
# safe here because appends are sequential (no dask parallelism).


import ee
import geedim
import gc
import warnings
import time
import fsspec
import argparse
import sys

# Suppress geedim size warnings and Zarr V3 consolidation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="geedim")
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
)

# Output chunk sizes.
# 120 × 120 spatial aligns with global_bias_adjustment.py's CHUNK_SIZE = 120.
# 120 × 120 × 500 × 8 bytes ≈ 57.6 MB per chunk (within the 5–100 MB sweet spot).
OUT_T_CHUNK = 500
OUT_S_CHUNK = 120


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land daily data and save to GCS Zarr."
    )
    parser.add_argument(
        "-v",
        "--variable",
        required=True,
        help="The ERA5-Land band/variable name (e.g., temperature_2m_min)",
    )
    args = parser.parse_args()

    band = args.variable

    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception:
        print(
            "Run 'uv run earthengine authenticate --auth_mode notebook' in your terminal first."
        )
        sys.exit(1)

    coll_id = "ECMWF/ERA5_LAND/DAILY_AGGR"
    years = list(range(1970, 2026))
    bucket = "clim_data_reg_useast1"

    # Final GCS Zarr path
    gcs_zarr_path = f"gs://{bucket}/era5_land/daily_aggregates/{band}.zarr"

    # ******
    old_zarr_path = f"gs://{bucket}/era5_land/daily_aggregates/{band}_old.zarr"

    # Check for existing stores and handle versioning before starting the download.
    # - If both exist: a previous backup is already in place; abort to avoid overwriting it.
    # - If only the original exists: back it up as _old, then proceed with a fresh download.
    # - Otherwise: proceed normally.
    fs = fsspec.filesystem("gs")
    orig_exists = fs.exists(gcs_zarr_path)
    old_exists = fs.exists(old_zarr_path)

    if orig_exists and old_exists:
        print(
            f"Both stores already exist:\n"
            f"  {gcs_zarr_path}\n"
            f"  {old_zarr_path}\n"
            f"Delete or rename one of them before rerunning. Exiting."
        )
        sys.exit(0)
    elif orig_exists and not old_exists:
        print(f"Existing store found. Renaming to _old before fresh download...")
        print(f"  {gcs_zarr_path} → {old_zarr_path}")
        fs.rename(gcs_zarr_path, old_zarr_path, recursive=True)
        print("Rename complete. Proceeding with fresh download.")

    # ******

    # Loop starting from the second year (index 1) since the first is processed
    for year in years:
        year_start_time = time.time()
        start_date = f"{year}-01-01"
        end_date = (
            f"{year + 1}-01-01"  # filterDate is end-exclusive; this captures Dec 31
        )

        print(f"--- Processing {year} ---")

        # Search and create collection
        coll = ee.ImageCollection(coll_id).filterDate(start_date, end_date).select(band)

        # Download collection to an xarray dataset in memory using High-Volume API
        print(f"Downloading {year} data to memory...")
        download_start_time = time.time()
        # ds = coll.gd.toXarray(max_tile_size=16, max_requests=2)
        ds = coll.gd.toXarray(
            max_tile_size=16, masked=True
        )  # masked=True → no-data pixels become NaN instead of 0
        download_duration = time.time() - download_start_time
        print(f"Download complete in: {download_duration / 60:.2f} minutes")

        # Direct upload/append to GCS Zarr
        print(f"Streaming {year} to GCS Zarr")
        upload_start_time = time.time()

        if year == years[0]:
            # For the first year, define the chunk grid on disk via encoding.
            chunk_dict = {
                "time": OUT_T_CHUNK,
                "y": OUT_S_CHUNK,
                "x": OUT_S_CHUNK,
                "latitude": OUT_S_CHUNK,
                "longitude": OUT_S_CHUNK,
            }
            # Get actual dimension names from the dataset and map them
            chunk_tuple = tuple(chunk_dict.get(d, 1) for d in ds[band].dims)
            encoding = {band: {"chunks": chunk_tuple}}
            ds.to_zarr(gcs_zarr_path, mode="w", safe_chunks=False, encoding=encoding)
        else:
            # When appending, Zarr already knows the chunk grid from the existing store.
            ds.to_zarr(gcs_zarr_path, mode="a", append_dim="time", safe_chunks=False)

        upload_duration = time.time() - upload_start_time
        print(f"Upload complete in: {upload_duration / 60:.2f} minutes")

        # CRITICAL: Force memory release
        print(f"Releasing memory for {year}...")
        del ds
        del coll

        # Clear fsspec/gcsfs instance caches which cause hidden leaks in loops
        fsspec.AbstractFileSystem.clear_instance_cache()

        # Run garbage collection twice to ensure circular references are caught
        gc.collect()
        gc.collect()

        year_duration = time.time() - year_start_time
        print(
            f"--- {year} Total Processing Time: {year_duration / 60:.2f} minutes ---\n"
        )

    print("Done.")


if __name__ == "__main__":
    main()
