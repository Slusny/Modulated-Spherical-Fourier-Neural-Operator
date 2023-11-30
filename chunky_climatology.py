import xarray as xr
import os
import numpy as np
import sys

from multiprocessing import Pool

basePath = "/mnt/qb/goswami/data/era5"
savepath = "/mnt/qb/work2/goswami0/gkd965/climate/chunky"

ds = xr.open_mfdataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_????.nc"),parallel=True)

ds['hourofyear'] = xr.DataArray(ds.time.dt.strftime('%m-%d %H'), coords=ds.time.coords)

# for lat in range(ds.dimensions.latitude):
#     for long in range(ds.dimensions.longitude):
#         ds_lat_long = ds.isel(latitude=lat,longitude=long)
#         ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
#         savefile = os.path.join(savepath,f"10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
#         ds_hourofyear.to_netcdf(savefile)

work = [(lat,long) for lat in range(ds.dimensions.latitude) for long in range(ds.dimensions.longitude)]

def calc_mean(lat,long):
    process = os.getpid()
    print(f"Process {process} works on {lat}-{long}")
    ds_lat_long = ds.isel(latitude=lat,longitude=long)
    ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
    savefile = os.path.join(savepath,f"10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
    ds_hourofyear.to_netcdf(savefile)


if __name__ == '__main__':
    with Pool(sys.argv[2]) as p:
        p.map(calc_mean, work)