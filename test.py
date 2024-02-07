import xarray as xr
import numpy as np
import pandas as pd

# temp = 15 + 8 * np.random.randn(2, 2, 3)
# time = pd.date_range("2014-09-06", periods=3)
# precip = 10 * np.random.rand(2, 2, 3)

# lon = [[-99.83, -99.32], [-99.79, -99.23]]
# lon_temp = [[-99.83, -99.32], [-99.79, -99.23]]
# lat = [[42.25, 42.21], [42.63, 42.59]]
# lat_temp = [[42.25, 42.21], [42.63, 42.51 ]]

# precip_array = xr.DataArray(precip,dims=["x","y","time"],coords=dict(
#         lon=(["x", "y"], lon),
#         lat=(["x", "y"], lat),
#         time=time))

# temp_array2 = xr.DataArray(temp,dims=["x","y","time"],coords=dict(
#         lon=(["x", "y"], lon_temp),
#         lat=(["x", "y"], lat_temp),
#         time=time))

# temp_array = xr.DataArray(temp,dims=["x","y","time"],coords=dict(
#         lon=(["x", "y"], lon),
#         lat=(["x", "y"], lat),
#         time=time))


# # d = xr.Dataset({"temp_data":temp_array,"precip_data":precip_array})
# # d = xr.Dataset()
# # d.assign(temperature=temp_array)
# # d.assign(pre=precip_array)
# # print(d)

# d = xr.Dataset(data_vars={"temp":temp_array,"precip":precip_array},coords=dict(
#         lon1=(["x", "y"], lon),
#         lat1=(["x", "y"], lat),
#         time=time))

# d.info()

# d2 = xr.Dataset(data_vars={"temp":temp_array,"precip":precip_array},coords=dict(
#         lon=(["x", "y"], lon_temp),
#         lat=(["x", "y"], lat_temp),
#         time=time))

# d2.info()
# print("done")

class A():
    def __init__(self):
        self.n_lat = 721
        self.n_lon = 1440
        self.hour_steps = 6
        self.input_type = "test"

        self.backbone_channels = 2

        self.checkpoint_path = "weights.tar"

class B(A):
    def __init__(self):
        super().__init__()
        self.hour_steps = 12
        self.test = self.n_lat * 3

b = B()
print(b.test)