import xarray as xr
import glob
import os

base_path = "/mnt/qb/goswami/data/era5"
# nc_files = list(sorted(glob.glob(os.path.join(base_path,"sst_1_deg/*.nc"))))

# dn = xr.open_dataset(nc_files[0])
# size= len(nc_files)
# print_percent = 1.
# print("start nc to zarr")
# for idx,file in enumerate(nc_files[1:]):
#     percent = (idx/size*100)
#     if percent > print_percent:
#         print(print_percent,"%")
#         print_percent += 1.0
#     ds = xr.open_dataset(file)
#     dn = xr.concat([dn, ds], dim='time')

# dn = dn.chunk(chunks={"time":1})
# dn.to_zarr(os.path.join(base_path, 'sst_1_deg.zarr'))

era = xr.open_zarr('/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')

dn = era["sea_surface_temperature"].coarsen(latitude=4,longitude=4,boundary='trim').mean()
dn = dn.chunk(chunks={"time":1})
dn.to_zarr(os.path.join(base_path, 'sst_1_deg.zarr'))