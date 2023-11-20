import xarray as xr
import matplotlib.pyplot as plt
from atmodata.datasets import ERA5
from torch.utils.data import DataLoader
# import metview as mv

# ds = (mv.read("ERA5_levels.grib")).to_dataset()

# ds = xr.load_dataset("/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/SFNO_HalfYearForecast_20231116-1441.grib", engine="cfgrib")
# print(ds.info())

path = "/mnt/qb/goswami/data/era5"
variables = ['z500', 't500']
years = [2022]
era5 = ERA5(path,variables,years)
data = DataLoader(dataset=era5, batch_size=1, shuffle=False)
first = next(iter(data))
print(first)
print(first.shape)