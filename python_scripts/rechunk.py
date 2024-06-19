import xarray as xr
import zarr
import dask.array as dsa
from rechunker import rechunk
from dask.diagnostics import ProgressBar
import os


# rechunk uv

# uv = "u"
# print(uv,flush=True)

# base_path = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440"
# save_path = "/mnt/qb/goswami/data/era5/weatherbench2/"
# if uv == "u":
#     uv100="u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
# else:
#     uv100="v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"


# uv100_new=uv+"100m_2_chunked_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
# uv100_temp = uv+"100m_2_chunked_temp.zarr"

# ds =  xr.open_zarr(os.path.join(base_path, uv100))

# target_chunks = {
#     uv+"100": {"time": 1, "latitude": 721, "longitude": 1440},
#     "time": None,  # don't rechunk this array
#     "latitude": None,
#     "longitude": None,
# }
# max_mem = "390GB"

# print("started rechunking",flush=True)
# target_store = os.path.join(save_path,uv100_new)
# temp_store = os.path.join(save_path,uv100_temp)
# source_array = ds
# array_plan = rechunk(
#     source_array, target_chunks, max_mem, target_store, temp_store=temp_store
# )

# print(array_plan)
# # result = array_plan.execute()


# with ProgressBar():
#     array_plan.execute()

#rechunk era5

base_path = "/mnt/qb/goswami/data/era5/weatherbench2/"
save_path = "/mnt/qb/goswami/data/era5/weatherbench2/"

era5="1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"


era5_new="1959-2023_01_10-wb13-6h-1440x721_sfno_chunked.zarr"
era5_temp ="sfno_temp.zarr"

ds =  xr.open_zarr(os.path.join(base_path, era5))
param_sfc_ERA5 = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure", "mean_sea_level_pressure", "total_column_water_vapour","u_component_of_wind","v_component_of_wind","geopotential","temperature","relative_humidity","sea_surface_temperature"]

ds = ds[param_sfc_ERA5]

target_chunks = {"time": 1}
max_mem = "1GB"

print("started rechunking",flush=True)
target_store = os.path.join(save_path,era5_new)
temp_store = os.path.join(save_path,era5_temp)
source_array = ds
array_plan = rechunk(
    source_array, target_chunks, max_mem, target_store, temp_store=temp_store
)

print(array_plan)
# result = array_plan.execute()


with ProgressBar():
    array_plan.execute()


# uv = "u"

# base_path = "/mnt/qb/goswami/data/era5/weatherbench2/"
# # base_path = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440"
# uv100="u100_smallset_test.zarr"

# uv100_new="u100_chunked_smallset_test.zarr"
# uv100_temp = "u100_temp_smallset_test.zarr"

# ds =  xr.open_zarr(os.path.join(base_path, uv100))

# target_chunks = {
#     uv+"100": {"time": 1, "latitude": 721, "longitude": 1440},
#     "time": None,  # don't rechunk this array
#     "latitude": None,
#     "longitude": None,
# }
# max_mem = "100GB"

# target_store = os.path.join(base_path,uv100_new)
# temp_store = os.path.join(base_path,uv100_temp)
# source_array = ds
# array_plan = rechunk(
#     source_array, target_chunks, max_mem, target_store, temp_store=temp_store
# )

# print(array_plan)
# result = array_plan.execute()


# with ProgressBar():
#     array_plan.execute()