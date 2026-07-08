
### tasmax and tasmin:
- Download `temperature_2m_max` and `temperature_2m_min` from ERA5-Land
- Calculate `tasrange` and `tasskew` (pre_generate_tasskew...py)
- Downscale `tasrange` and `tasskew`
- Reconstruct `tasmax` and `tasmin` (post_tas_extr...py)

### wetbulbmax:
- Download `temperature_2m_max` and `dewpoint_temperature_min` from ERA5-Land
- Calculate `hursmin` (`pre_generate_hursmin.py`)
- Downscale `tasmax` (already done -- see "tasmax and tasmin" section)
- Downscale `hursmin`, calculating CMIP6's `hursmin` on the fly with `tasmax` and `tasmin`
    - `tasmin` is used here as a replacement of dewpoint (and min dewpoint) in August-Roche
- Calculate `wetbulbmax` with Stull formulation (`post_max_wetbulb.py`)