# SCRIPT TO DOWNLOAD ERA5-LAND DAILY DATA
# RUNNING THIS SCRIPT REQUIRES ~60 GB OF MEMORY

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
            "Please run 'uv run earthengine authenticate --auth_mode notebook' in your terminal first."
        )
        sys.exit(1)

    coll_id = "ECMWF/ERA5_LAND/DAILY_AGGR"
    years = list(range(1970, 2026))
    bucket = "clim_data_reg_useast1"

    # Final GCS Zarr path
    gcs_zarr_path = f"gs://{bucket}/era5_land/daily_aggregates/{band}.zarr"

    # Loop starting from the second year (index 1) since the first is processed
    for year in years:
        year_start_time = time.time()
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        print(f"--- Processing {year} ---")

        # Search and create collection
        coll = ee.ImageCollection(coll_id).filterDate(start_date, end_date).select(band)

        # Download collection to an xarray dataset in memory using High-Volume API
        print(f"Downloading {year} data to memory...")
        download_start_time = time.time()
        ds = coll.gd.toXarray(max_tile_size=16)
        download_duration = time.time() - download_start_time
        print(f"Download complete in: {download_duration / 60:.2f} minutes")

        # Direct upload/append to GCS Zarr
        print(f"Streaming {year} to GCS Zarr")
        upload_start_time = time.time()

        if year == years[0]:
            # For the first year, define the chunk grid on disk via encoding.
            chunk_dict = {
                "time": ds.sizes["time"],
                "y": 200,
                "x": 200,
                "latitude": 200,
                "longitude": 200,
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
