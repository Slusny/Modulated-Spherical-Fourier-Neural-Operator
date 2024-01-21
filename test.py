import xarray as xr
import numpy as np
import pandas as pd

temp = 15 + 8 * np.random.randn(2, 3, 3)
time = pd.date_range("2014-09-06", periods=3)
precip = 10 * np.random.rand(2, 2, 3)

lon = [[-99.83, -99.32], [-99.79, -99.23]]
lon_temp = [[-99.83, -99.32,-90], [-99.79, -99.23,-90]]
lat = [[42.25, 42.21], [42.63, 42.59]]
lat_temp = [[42.25, 42.21,41], [42.63, 42.59,41 ]]

precip_array = xr.DataArray(precip,dims=["x","yy","time"],coords=dict(
        lon1=(["x", "yy"], lon),
        lat1=(["x", "yy"], lat),
        time=time))

temp_array = xr.DataArray(temp,dims=["x","y","time"],coords=dict(
        lon=(["x", "y"], lon_temp),
        lat=(["x", "y"], lat_temp),
        time=time))


# d = xr.Dataset({"temp_data":temp_array,"precip_data":precip_array})
d = xr.Dataset()
d.assign(temperature=temp_array)
d.assign(pre=precip_array)
print(d)