"""
The relative humidity has to be downloaded from Copernicus to have the right format for the pretrained SFNO model.
Copernicus generates single netcdf files for each year.
This script allows to allocate all these .nc files into a single zar-file, simplifying the handling of relative humidity data. 
"""

import xarray as xr
import glob
import os

def combine_relative_humidity():
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    datasets = []
    for level in levels:
        # print(f"/mnt/qb/goswami/data/era5/multi_pressure_level/relative_humidity/{level}/relative_humidity_1979_*.nc")
        ds = xr.open_mfdataset(f"/mnt/qb/goswami/data/era5/multi_pressure_level/relative_humidity/{level}/relative_humidity_*.nc",chunks={'time':1,'level':13,'longitude':1440,'latitude':721})
        level_ = xr.DataArray([level],[('level',[level])])
        datasets.append(ds.expand_dims(level=level_))
    return xr.concat(datasets,dim="level")


base_path = "/mnt/qb/goswami/data/era5"
print('save zarr')
r = combine_relative_humidity()
print("loaded relative humidity")
zarr_save_path = os.path.join(base_path, 'relative_humidity_1979_to_2018.zarr')
for idx, year in enumerate(range(1979,2019)):
    print(f"loading {year}")
    x = r.sel(time=slice(f"{year}-01-01",f"{year}-12-31"))['r']
    arr = xr.DataArray(x.to_numpy(),
        dims=["level","time","latitude","longitude"],
        coords=dict(
            level=("level",x.level.to_numpy()),
            time=("time",x.time.to_numpy()),
            latitude=("latitude", x.latitude.to_numpy()),
            longitude=("longitude", x.longitude.to_numpy()),
        )
    )
    if idx == 0:
        xr.Dataset(data_vars={'r':arr}).chunk({'time': 1, 'level':13, 'latitude':721,'longitude':1440}).to_zarr(zarr_save_path)
    else:
        xr.Dataset(data_vars={'r':arr}).chunk({'time': 1, 'level':13, 'latitude':721,'longitude':1440}).to_zarr(zarr_save_path,mode="a",append_dim="time")
        
print("done")