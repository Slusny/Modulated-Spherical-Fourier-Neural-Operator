import xarray as xr
import os
import numpy as np
import sys

basePath = "/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/"
file = "created_2023-11-27-2312_SFNO_HalfYearForecast_startfrom_2022-01-01.grib"
file_path = os.path.join(basePath,file)
s_sfno = xr.open_dataset(file_path, engine="cfgrib", chunks={"step": 4})

basePath = "/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/"
file = "v10_SFNO_HalfYearForecast_startfrom_2022-01-01.nc"
file_path = os.path.join(basePath,file)
v_sfno = xr.open_dataset(file_path, chunks={"step": 4})
