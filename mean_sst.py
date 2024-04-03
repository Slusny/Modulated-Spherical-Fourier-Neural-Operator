import xarray as xr
import os
import numpy as np

ds = xr.open_dataset("/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
sst = ds["sea_surface_temperature"]

save_path = "/mnt/qb/work2/goswami0/gkd965/Assets/sfno"

mean = sst.mean()
print("mean sst: ", mean)
np.save(os.path.join(save_path,"global_means_sst8.npy"), mean)
std = sst.std()
print("std sst: ", std)
np.save(os.path.join(save_path,"global_std_sst8.npy"), std)
print("done")

