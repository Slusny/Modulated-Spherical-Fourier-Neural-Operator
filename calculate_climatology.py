import xarray as xr
import os
import numpy as np
import sys

path = sys.argv[0] #"/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/"
file = sys.argv[1] #"era5_"

file_path = os.path.join(path,file)
print('file_path: ', file_path)

ds = xr.open_dataset(file_path)
print(ds.info())

# ds_sfno.groupby("step.days").mean(dim="step")
# ds.to_netcdf("/home/gkd965/Files/ERA5_climatology.nc")