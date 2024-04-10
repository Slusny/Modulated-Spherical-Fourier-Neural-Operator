import xarray as xr
import zarr
import dask.array as dsa
from rechunker import rechunk
from dask.diagnostics import ProgressBar
import os

uv = "u"

base_path = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440"
uv100=uv+"100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"

uv100_new=uv+"100m_chunked_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
uv100_temp = uv+"100m_chunked_temp.zarr"

ds =  xr.open_zarr(os.path.join(base_path, uv100))

target_chunks = {
    uv+"100": {"time": 1, "latitude": 721, "longitude": 1440},
    "time": None,  # don't rechunk this array
    "latitude": None,
    "longitude": None,
}
max_mem = "300GB"

target_store = os.path.join(base_path,uv100_new)
temp_store = os.path.join(base_path,uv100_temp)
source_array = ds[uv+"100"]
array_plan = rechunk(
    source_array, target_chunks, max_mem, target_store, temp_store=temp_store
)

print(array_plan)
result = array_plan.execute()


with ProgressBar():
    array_plan.execute()