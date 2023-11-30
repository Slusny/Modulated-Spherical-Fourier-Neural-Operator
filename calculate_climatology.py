import xarray as xr
import os
import numpy as np
import sys

# path = sys.argv[1] #"/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/"
# file = sys.argv[2] #"20231108.grib"

# file_path = os.path.join(path,file)
# print('file_path: ', file_path)

basePath = "/mnt/qb/goswami/data/era5"

param_sfc = [
    "10m_u_component_of_wind", #from 1979 available, + 1979_2021 combined
    "10m_v_component_of_wind", #from 1979 available, + 1979_2021 combined
    "2m_temperature",          #from 1959 available, + 1959_2021 combined
    "surface_pressure", 
    "msl", 
    "total_column_water_vapour", 
    "100u", 
    "100v"
]

combined_years = {
    "10m_u_component_of_wind": "10m_u_component_of_wind_sfc_1979_2021.nc"
}

param_level_pl = (["t", "u", "v", "z", "r"], [1000, 850, 500, 250, 50])

assets_extra_dir = "0.1"

ordering = [
    "10u",
    "10v",
    "2t",
    "sp",
    "msl",
    "t850",
    "u1000",
    "v1000",
    "z1000",
    "u850",
    "v850",
    "z850",
    "u500",
    "v500",
    "z500",
    "t500",
    "z50",
    "r500",
    "r850",
    "tcwv",
    "100u",
    "100v",
    "u250",
    "v250",
    "z250",
    "t250",
]

paths = []
# for sfc in param_sfc:
#     paths.append(os.path.join(basePath, 'single_pressure_level', sfc, "*.nc"))
# ds = xr.open_mfdataset(paths, concat_dim="time", preprocess=partial_func) 

# sfc = "2m_temperature/2m_temperature_sfc_1959_2021.nc"
# path = os.path.join(basePath, 'single_pressure_level', sfc)
print("loading data")
# ds = xr.open_mfdataset(os.path.join(basePath, 'single_pressure_level', '2m_temperature', "2m_temperature_????.nc"),parallel=True)#, concat_dim="time", combine="nested", preprocess=partial_func)) 
ds = xr.open_mfdataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_????.nc"),parallel=True)
print("ds: ")
print(ds.info())
print("fin loading")

ds['hourofyear'] = xr.DataArray(ds.time.dt.strftime('%m-%d %H'), coords=ds.time.coords)
ds_hourofyear = ds.groupby("hourofyear")
mean_temps = ds_hourofyear.mean()

print("current working directory: ", os.getcwd())
savepath = "/mnt/qb/work2/goswami0/gkd965/climate/t2m_1959-2021_hourofyear_mean.nc"
# savepath = "/home/goswami/gkd965/t2m_1959-2021_hourofyear_mean.nc"
mean_temps.to_netcdf(savepath)


# ds_grouped = ds.groupby("time.dayofyear")
# print("\nds_grouped: ")
# print(ds_grouped.groups)
# print(list(ds_grouped))

# ds_resampled = ds_grouped.resample(time='6H')
# print("\nds_resampled: ")
# print(ds_resampled.info())

# ds_grouped = ds.groupby("time.month").mean(dim="time")


# ds = xr.open_dataset()
# print(ds.info())

# ds_sfno.groupby("step.days").mean(dim="step")
# ds.to_netcdf("/home/gkd965/Files/ERA5_climatology.nc")