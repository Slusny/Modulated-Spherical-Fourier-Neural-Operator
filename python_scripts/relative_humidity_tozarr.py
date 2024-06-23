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
        

# r.to_zarr(zarr_save_path,mode='w')
print("done")