import xarray as xr

data = xr.open_dataset('/mnt/qb/work2/goswami0/gkd965/outputs/sfno/leadtime_8760_20231213-1809.grib', chunks={'step': 1})
data.to_netcdf('/mnt/qb/work2/goswami0/gkd965/outputs/sfno/leadtime_8760_20231213-1809.nc', unlimited_dims='time')