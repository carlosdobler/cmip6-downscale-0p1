# SCRIPT TO DOWNLOAD ERA5-LAND DAILY DATA
# RUNNING THIS SCRIPT REQUIRES ~60 GB OF MEMORY
#
# Time chunks are set to OUT_T_CHUNK = 500 days on first write (~57.6 MB at 120×120 spatial).
# Spatial chunks of 120×120 align with global_bias_adjustment.py's CHUNK_SIZE = 120.
# Subsequent yearly appends use safe_chunks=False because each year's data crosses the
# 500-day chunk boundary, requiring a read-modify-write on the last partial chunk. This is
# safe here because appends are sequential (no dask parallelism).
#
# --replace / --resume:
# The main download loop only calls `to_zarr` after a full year has been successfully
# downloaded into memory, so a store never contains a partially-written year (interruptions,
# e.g. running out of GEE credits, happen during the download step, not the write). This means
# `--resume` can safely trust the last year found in the store and continue from the next one.


import ee
import geedim
import gc
import warnings
import time
import fsspec
import xarray as xr
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

COLL_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
YEARS = list(range(1970, 2026))
BUCKET = "clim_data_reg_useast1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land daily data and save to GCS Zarr."
    )
    parser.add_argument(
        "-v",
        "--variable",
        required=True,
        help="The ERA5-Land band/variable name (e.g., temperature_2m_min)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--replace",
        action="store_true",
        help=(
            "If the destination store already exists, back it up to "
            "'{band}_old.zarr' and start a fresh download from scratch."
        ),
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted download: scan the existing store for the "
            "last saved year and continue downloading from the following year."
        ),
    )
    return parser.parse_args()


def init_earth_engine():
    try:
        ee.Initialize()
    except Exception:
        print(
            "Run 'uv run earthengine authenticate --auth_mode notebook' in your terminal first."
        )
        sys.exit(1)


def backup_existing_store(fs, gcs_zarr_path, old_zarr_path):
    """Rename the current store to its single `_old` backup slot."""
    if fs.exists(old_zarr_path):
        print(
            f"Backup slot already occupied:\n"
            f"  {old_zarr_path}\n"
            f"Delete or rename it before rerunning with --replace. Exiting."
        )
        sys.exit(1)
    print("Existing store found. Renaming to _old before fresh download...")
    print(f"  {gcs_zarr_path} -> {old_zarr_path}")
    fs.rename(gcs_zarr_path, old_zarr_path, recursive=True)
    print("Rename complete. Proceeding with fresh download.")


def get_resume_year(gcs_zarr_path):
    """Return the year to resume downloading from, based on the last year saved."""
    ds = xr.open_zarr(gcs_zarr_path)
    last_saved_year = int(ds["time"].dt.year.max())
    ds.close()
    print(f"Last saved year in store: {last_saved_year}")
    return last_saved_year + 1


def resolve_store_state(fs, gcs_zarr_path, old_zarr_path, replace, resume):
    """
    Decide the starting year and whether the first write should create a new
    store (mode="w") or append to an existing one (mode="a").

    Returns (start_year, is_fresh_store).
    """
    store_exists = fs.exists(gcs_zarr_path)

    if resume:
        if not store_exists:
            print(
                f"--resume was given but no store was found at:\n"
                f"  {gcs_zarr_path}\n"
                f"Nothing to resume. Exiting."
            )
            sys.exit(1)
        start_year = get_resume_year(gcs_zarr_path)
        if start_year > YEARS[-1]:
            print(f"Store already covers through {YEARS[-1]}. Nothing to do.")
            sys.exit(0)
        return start_year, False

    if replace:
        if store_exists:
            backup_existing_store(fs, gcs_zarr_path, old_zarr_path)
        return YEARS[0], True

    # Default: neither flag given.
    if store_exists:
        print(
            f"Store already exists:\n"
            f"  {gcs_zarr_path}\n"
            f"Use --replace to back it up and start over, or --resume to "
            f"continue an interrupted download. Exiting."
        )
        sys.exit(1)
    return YEARS[0], True


def download_year(band, year):
    """Download one year of a band from ERA5-Land into an in-memory xarray Dataset."""
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"  # filterDate is end-exclusive; this captures Dec 31

    coll = ee.ImageCollection(COLL_ID).filterDate(start_date, end_date).select(band)

    print(f"Downloading {year} data to memory...")
    download_start_time = time.time()
    ds = coll.gd.toXarray(
        max_tile_size=16, masked=True
    )  # masked=True -> no-data pixels become NaN instead of 0
    download_duration = time.time() - download_start_time
    print(f"Download complete in: {download_duration / 60:.2f} minutes")

    del coll
    return ds


def write_year(ds, band, gcs_zarr_path, is_fresh_store):
    """Write/append one year of data to the destination Zarr store."""
    print(f"Streaming to GCS Zarr")
    upload_start_time = time.time()

    if is_fresh_store:
        # For the first year of a fresh store, define the chunk grid on disk via encoding.
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


def release_memory():
    """
    Force memory release between year iterations.

    Must be called *after* the caller has already `del`eted its own references
    (e.g. `ds`) - deleting objects passed as arguments here would only unbind
    the local parameter name, not the caller's variable, leaving the object
    referenced (and therefore alive) until the caller's next reassignment.
    """
    # Clear fsspec/gcsfs instance caches which cause hidden leaks in loops
    fsspec.AbstractFileSystem.clear_instance_cache()
    # Run garbage collection twice to ensure circular references are caught
    gc.collect()
    gc.collect()


def main():
    args = parse_args()
    band = args.variable

    init_earth_engine()

    gcs_zarr_path = f"gs://{BUCKET}/era5_land/daily_aggregates/{band}.zarr"
    old_zarr_path = f"gs://{BUCKET}/era5_land/daily_aggregates/{band}_old.zarr"

    fs = fsspec.filesystem("gs")
    start_year, is_fresh_store = resolve_store_state(
        fs, gcs_zarr_path, old_zarr_path, args.replace, args.resume
    )

    start_index = YEARS.index(start_year)
    for year in YEARS[start_index:]:
        year_start_time = time.time()
        print(f"--- Processing {year} ---")

        ds = download_year(band, year)
        write_year(ds, band, gcs_zarr_path, is_fresh_store and year == start_year)

        print(f"Releasing memory for {year}...")
        del ds
        release_memory()

        year_duration = time.time() - year_start_time
        print(
            f"--- {year} Total Processing Time: {year_duration / 60:.2f} minutes ---\n"
        )

    print("Done.")


if __name__ == "__main__":
    main()
