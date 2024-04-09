# import xarray as xr
# import numpy as np
# import pandas as pd

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

# class A():
#     def __init__(self):
#         self.n_lat = 721
#         self.n_lon = 1440
#         self.hour_steps = 6
#         self.input_type = "test"

#         self.backbone_channels = 2

#         self.checkpoint_path = "weights.tar"

# class B(A):
#     def __init__(self):
#         super().__init__()
#         self.hour_steps = 12
#         self.test = self.n_lat * 3

# b = B()
# print(b.test)

# long = 1440
# lat = 721

# def kernel(j,i,w,h):
#     x_temp = i+w
#     y_temp = j+h
#     if x_temp < 0:
#         x = long + x_temp
#     elif x_temp >= long:
#         x = x_temp + long
#     else:
#         x = x_temp
        
#     if y_temp < 0:
#         y = lat + y_temp
#     elif y_temp >= lat:
#         y = y_temp + lat
#     else:
#         y = y_temp
#     return (y,x)

# a = kernel(0,1437,-1,-1)
# print(a)

# class test():
#     c = 3
#     def __init__(self):
#         self.a = 1
#         b = 2

#     def print(self):
#         print(self.a, self.c)
    
# t = test()
# t.print()

# import torch

# d = (torch.tensor([2,2]),torch.tensor([2,2]))
# for i in [d[0],d[1]]:
#     i.to("cuda:0")
# d[0].to("cuda:0")
# a,b = d
# a.to("cuda:0")
# # d[1].to("cuda:0")
# print(a.device)
# # print(d[1])
# print(torch.tensor([2,2]).to("cuda:0"))

# for a,b in d:
#     a.to("cuda:0")
#     b = b.to("cuda:0")
#     print(a.device)
#     print(b.device)

# for _ in [d[0],d[1]]:
#     _ = _.to("cuda:0")
# print(d[0].device)

# input, sst = d[0].to("cuda:0"), d[1].to("cuda:0")
# print(input.device)
# print(sst.device)

# class test():
#     def __init__(self,a=1,b=2,**kwargs):
#         self.a = {"a":a}
#         self.b = b
#     def p(self):
#         print(self.a,self.b)
#     def mod(self):
#         c = {"a": 100}
#         self.a = c
#         c["a"] = 200
#         return c

# t = test(a=3,b=4)
# t.p()
# t.mod()
# t.p()

trained = [ pos_embed,
    encoder.fwd.0.weight,
    encoder.fwd.0.bias,
    encoder.fwd.2.weight,
    blocks.0.norm0.weight,
    blocks.0.norm0.bias,
    blocks.0.filter_layer.filter.wout,
    blocks.0.filter_layer.filter.w.0,
    blocks.0.filter_layer.filter.w.1,
    blocks.0.filter_layer.filter.w.2,
    blocks.0.filter_layer.filter.activation.bias,
    blocks.0.norm1.weight,
    blocks.0.norm1.bias,
    blocks.0.mlp.fwd.0.weight,
    blocks.0.mlp.fwd.0.bias,
    blocks.0.mlp.fwd.2.weight,
    blocks.0.mlp.fwd.2.bias,
    blocks.1.norm0.weight,
    blocks.1.norm0.bias,
    blocks.1.filter_layer.filter.wout,
    blocks.1.filter_layer.filter.w.0,
    blocks.1.filter_layer.filter.w.1,
    blocks.1.filter_layer.filter.w.2,
    blocks.1.filter_layer.filter.activation.bias,
    blocks.1.inner_skip.weight,
    blocks.1.inner_skip.bias,
    blocks.1.norm1.weight,
    blocks.1.norm1.bias,
    blocks.1.mlp.fwd.0.weight,
    blocks.1.mlp.fwd.0.bias,
    blocks.1.mlp.fwd.2.weight,
    blocks.1.mlp.fwd.2.bias,
    blocks.2.norm0.weight,
    blocks.2.norm0.bias,
    blocks.2.filter_layer.filter.wout,
    blocks.2.filter_layer.filter.w.0,
    blocks.2.filter_layer.filter.w.1,
    blocks.2.filter_layer.filter.w.2,
    blocks.2.filter_layer.filter.activation.bias,
    blocks.2.inner_skip.weight,
    blocks.2.inner_skip.bias,
    blocks.2.norm1.weight,
    blocks.2.norm1.bias,
    blocks.2.mlp.fwd.0.weight,
    blocks.2.mlp.fwd.0.bias,
    blocks.2.mlp.fwd.2.weight,
    blocks.2.mlp.fwd.2.bias,
    blocks.3.norm0.weight,
    blocks.3.norm0.bias,
    blocks.3.filter_layer.filter.wout,
    blocks.3.filter_layer.filter.w.0,
    blocks.3.filter_layer.filter.w.1,
    blocks.3.filter_layer.filter.w.2,
    blocks.3.filter_layer.filter.activation.bias,
    blocks.3.inner_skip.weight,
    blocks.3.inner_skip.bias,
    blocks.3.norm1.weight,
    blocks.3.norm1.bias,
    blocks.3.mlp.fwd.0.weight,
    blocks.3.mlp.fwd.0.bias,
    blocks.3.mlp.fwd.2.weight,
    blocks.3.mlp.fwd.2.bias,
    blocks.4.norm0.weight,
    blocks.4.norm0.bias,
    blocks.4.filter_layer.filter.wout,
    blocks.4.filter_layer.filter.w.0,
    blocks.4.filter_layer.filter.w.1,
    blocks.4.filter_layer.filter.w.2,
    blocks.4.filter_layer.filter.activation.bias,
    blocks.4.inner_skip.weight,
    blocks.4.inner_skip.bias,
    blocks.4.norm1.weight,
    blocks.4.norm1.bias,
    blocks.4.mlp.fwd.0.weight,
    blocks.4.mlp.fwd.0.bias,
    blocks.4.mlp.fwd.2.weight,
    blocks.4.mlp.fwd.2.bias,
    blocks.5.norm0.weight,
    blocks.5.norm0.bias,
    blocks.5.filter_layer.filter.wout,
    blocks.5.filter_layer.filter.w.0,
    blocks.5.filter_layer.filter.w.1,
    blocks.5.filter_layer.filter.w.2,
    blocks.5.filter_layer.filter.activation.bias,
    blocks.5.inner_skip.weight,
    blocks.5.inner_skip.bias,
    blocks.5.norm1.weight,
    blocks.5.norm1.bias,
    blocks.5.mlp.fwd.0.weight,
    blocks.5.mlp.fwd.0.bias,
    blocks.5.mlp.fwd.2.weight,
    blocks.5.mlp.fwd.2.bias,
    blocks.6.norm0.weight,
    blocks.6.norm0.bias,
    blocks.6.filter_layer.filter.wout,
    blocks.6.filter_layer.filter.w.0,
    blocks.6.filter_layer.filter.w.1,
    blocks.6.filter_layer.filter.w.2,
    blocks.6.filter_layer.filter.activation.bias,
    blocks.6.inner_skip.weight,
    blocks.6.inner_skip.bias,
    blocks.6.norm1.weight,
    blocks.6.norm1.bias,
    blocks.6.mlp.fwd.0.weight,
    blocks.6.mlp.fwd.0.bias,
    blocks.6.mlp.fwd.2.weight,
    blocks.6.mlp.fwd.2.bias,
    blocks.7.norm0.weight,
    blocks.7.norm0.bias,
    blocks.7.filter_layer.filter.wout,
    blocks.7.filter_layer.filter.w.0,
    blocks.7.filter_layer.filter.w.1,
    blocks.7.filter_layer.filter.w.2,
    blocks.7.filter_layer.filter.activation.bias,
    blocks.7.inner_skip.weight,
    blocks.7.inner_skip.bias,
    blocks.7.norm1.weight,
    blocks.7.norm1.bias,
    blocks.7.mlp.fwd.0.weight,
    blocks.7.mlp.fwd.0.bias,
    blocks.7.mlp.fwd.2.weight,
    blocks.7.mlp.fwd.2.bias,
    blocks.8.norm0.weight,
    blocks.8.norm0.bias,
    blocks.8.filter_layer.filter.wout,
    blocks.8.filter_layer.filter.w.0,
    blocks.8.filter_layer.filter.w.1,
    blocks.8.filter_layer.filter.w.2,
    blocks.8.filter_layer.filter.activation.bias,
    blocks.8.inner_skip.weight
    blocks.8.inner_skip.bias
    blocks.8.norm1.weight
    blocks.8.norm1.bias
    blocks.8.mlp.fwd.0.weight
    blocks.8.mlp.fwd.0.bias
    blocks.8.mlp.fwd.2.weight
    blocks.8.mlp.fwd.2.bias
    blocks.9.norm0.weight
    blocks.9.norm0.bias
    blocks.9.filter_layer.filter.wout
    blocks.9.filter_layer.filter.w.0
    blocks.9.filter_layer.filter.w.1
    blocks.9.filter_layer.filter.w.2
    blocks.9.filter_layer.filter.activation.bias
    blocks.9.inner_skip.weight,
    blocks.9.inner_skip.bias,
    blocks.9.norm1.weight,
    blocks.9.norm1.bias,
    blocks.9.mlp.fwd.0.weight,
    blocks.9.mlp.fwd.0.bias,
    blocks.9.mlp.fwd.2.weight,
    blocks.9.mlp.fwd.2.bias,
    blocks.10.norm0.weight,
    blocks.10.norm0.bias,
    blocks.10.filter_layer.filter.wout,
    blocks.10.filter_layer.filter.w.0,
    blocks.10.filter_layer.filter.w.1,
    blocks.10.filter_layer.filter.w.2,
    blocks.10.filter_layer.filter.activation.bias,
    blocks.10.inner_skip.weight,
    blocks.10.inner_skip.bias,
    blocks.10.norm1.weight,
    blocks.10.norm1.bias,
    blocks.10.mlp.fwd.0.weight,
    blocks.10.mlp.fwd.0.bias,
    blocks.10.mlp.fwd.2.weight,
    blocks.10.mlp.fwd.2.bias,
    blocks.11.norm0.weight,
    blocks.11.norm0.bias,
    blocks.11.filter_layer.filter.wout,
    blocks.11.filter_layer.filter.w.0,
    blocks.11.filter_layer.filter.w.1,
    blocks.11.filter_layer.filter.w.2,
    blocks.11.filter_layer.filter.activation.bias,
    blocks.11.norm1.weight,
    blocks.11.norm1.bias,
    decoder.fwd.0.weight,
    decoder.fwd.0.bias,
    decoder.fwd.2.weight,
]


orig = [ pos_embed ,
    encoder.fwd.0.weight ,
    encoder.fwd.0.bias ,
    encoder.fwd.2.weight ,
    blocks.0.norm0.weight ,
    blocks.0.norm0.bias ,
    blocks.0.filter_layer.filter.wout ,
    blocks.0.filter_layer.filter.w.0 ,
    blocks.0.filter_layer.filter.w.1 ,
    blocks.0.filter_layer.filter.w.2 ,
    blocks.0.filter_layer.filter.activation.bias ,
    blocks.0.norm1.weight ,
    blocks.0.norm1.bias ,
    blocks.0.mlp.fwd.0.weight ,
    blocks.0.mlp.fwd.0.bias ,
    blocks.0.mlp.fwd.2.weight ,
    blocks.0.mlp.fwd.2.bias ,
    blocks.1.norm0.weight ,
    blocks.1.norm0.bias ,
    blocks.1.filter_layer.filter.wout ,
    blocks.1.filter_layer.filter.w.0 ,
    blocks.1.filter_layer.filter.w.1 ,
    blocks.1.filter_layer.filter.w.2 ,
    blocks.1.filter_layer.filter.activation.bias ,
    blocks.1.inner_skip.weight ,
    blocks.1.inner_skip.bias ,
    blocks.1.norm1.weight ,
    blocks.1.norm1.bias ,
    blocks.1.mlp.fwd.0.weight ,
    blocks.1.mlp.fwd.0.bias ,
    blocks.1.mlp.fwd.2.weight ,
    blocks.1.mlp.fwd.2.bias ,
    blocks.2.norm0.weight ,
    blocks.2.norm0.bias ,
    blocks.2.filter_layer.filter.wout ,
    blocks.2.filter_layer.filter.w.0 ,
    blocks.2.filter_layer.filter.w.1 ,
    blocks.2.filter_layer.filter.w.2 ,
    blocks.2.filter_layer.filter.activation.bias ,
    blocks.2.inner_skip.weight ,
    blocks.2.inner_skip.bias ,
    blocks.2.norm1.weight ,
    blocks.2.norm1.bias ,
    blocks.2.mlp.fwd.0.weight ,
    blocks.2.mlp.fwd.0.bias ,
    blocks.2.mlp.fwd.2.weight ,
    blocks.2.mlp.fwd.2.bias ,
    blocks.3.norm0.weight ,
    blocks.3.norm0.bias ,
    blocks.3.filter_layer.filter.wout ,
    blocks.3.filter_layer.filter.w.0 ,
    blocks.3.filter_layer.filter.w.1 ,
    blocks.3.filter_layer.filter.w.2 ,
    blocks.3.filter_layer.filter.activation.bias ,
    blocks.3.inner_skip.weight ,
    blocks.3.inner_skip.bias ,
    blocks.3.norm1.weight ,
    blocks.3.norm1.bias ,
    blocks.3.mlp.fwd.0.weight ,
    blocks.3.mlp.fwd.0.bias ,
    blocks.3.mlp.fwd.2.weight ,
    blocks.3.mlp.fwd.2.bias ,
    blocks.4.norm0.weight ,
    blocks.4.norm0.bias ,
    blocks.4.filter_layer.filter.wout ,
    blocks.4.filter_layer.filter.w.0 ,
    blocks.4.filter_layer.filter.w.1 ,
    blocks.4.filter_layer.filter.w.2 ,
    blocks.4.filter_layer.filter.activation.bias ,
    blocks.4.inner_skip.weight ,
    blocks.4.inner_skip.bias ,
    blocks.4.norm1.weight ,
    blocks.4.norm1.bias ,
    blocks.4.mlp.fwd.0.weight ,
    blocks.4.mlp.fwd.0.bias ,
    blocks.4.mlp.fwd.2.weight ,
    blocks.4.mlp.fwd.2.bias ,
    blocks.5.norm0.weight ,
    blocks.5.norm0.bias ,
    blocks.5.filter_layer.filter.wout ,
    blocks.5.filter_layer.filter.w.0 ,
    blocks.5.filter_layer.filter.w.1 ,
    blocks.5.filter_layer.filter.w.2 ,
    blocks.5.filter_layer.filter.activation.bias ,
    blocks.5.inner_skip.weight ,
    blocks.5.inner_skip.bias ,
    blocks.5.norm1.weight ,
    blocks.5.norm1.bias ,
    blocks.5.mlp.fwd.0.weight ,
    blocks.5.mlp.fwd.0.bias ,
    blocks.5.mlp.fwd.2.weight ,
    blocks.5.mlp.fwd.2.bias ,
    blocks.6.norm0.weight ,
    blocks.6.norm0.bias ,
    blocks.6.filter_layer.filter.wout ,
    blocks.6.filter_layer.filter.w.0 ,
    blocks.6.filter_layer.filter.w.1 ,
    blocks.6.filter_layer.filter.w.2 ,
    blocks.6.filter_layer.filter.activation.bias ,
    blocks.6.inner_skip.weight ,
    blocks.6.inner_skip.bias ,
    blocks.6.norm1.weight ,
    blocks.6.norm1.bias ,
    blocks.6.mlp.fwd.0.weight ,
    blocks.6.mlp.fwd.0.bias ,
    blocks.6.mlp.fwd.2.weight ,
    blocks.6.mlp.fwd.2.bias ,
    blocks.7.norm0.weight ,
    blocks.7.norm0.bias ,
    blocks.7.filter_layer.filter.wout ,
    blocks.7.filter_layer.filter.w.0 ,
    blocks.7.filter_layer.filter.w.1 ,
    blocks.7.filter_layer.filter.w.2 ,
    blocks.7.filter_layer.filter.activation.bias ,
    blocks.7.inner_skip.weight ,
    blocks.7.inner_skip.bias ,
    blocks.7.norm1.weight ,
    blocks.7.norm1.bias ,
    blocks.7.mlp.fwd.0.weight ,
    blocks.7.mlp.fwd.0.bias ,
    blocks.7.mlp.fwd.2.weight ,
    blocks.7.mlp.fwd.2.bias ,
    blocks.8.norm0.weight ,
    blocks.8.norm0.bias ,
    blocks.8.filter_layer.filter.wout ,
    blocks.8.filter_layer.filter.w.0 ,
    blocks.8.filter_layer.filter.w.1 ,
    blocks.8.filter_layer.filter.w.2 ,
    blocks.8.filter_layer.filter.activation.bias ,
    blocks.8.inner_skip.weight ,
    blocks.8.inner_skip.bias ,
    blocks.8.norm1.weight ,
    blocks.8.norm1.bias ,
    blocks.8.mlp.fwd.0.weight ,
    blocks.8.mlp.fwd.0.bias ,
    blocks.8.mlp.fwd.2.weight ,
    blocks.8.mlp.fwd.2.bias ,
    blocks.9.norm0.weight ,
    blocks.9.norm0.bias ,
    blocks.9.filter_layer.filter.wout ,
    blocks.9.filter_layer.filter.w.0 ,
    blocks.9.filter_layer.filter.w.1 ,
    blocks.9.filter_layer.filter.w.2 ,
    blocks.9.filter_layer.filter.activation.bias ,
    blocks.9.inner_skip.weight ,
    blocks.9.inner_skip.bias ,
    blocks.9.norm1.weight ,
    blocks.9.norm1.bias ,
    blocks.9.mlp.fwd.0.weight ,
    blocks.9.mlp.fwd.0.bias ,
    blocks.9.mlp.fwd.2.weight ,
    blocks.9.mlp.fwd.2.bias ,
    blocks.10.norm0.weight ,
    blocks.10.norm0.bias ,
    blocks.10.filter_layer.filter.wout ,
    blocks.10.filter_layer.filter.w.0 ,
    blocks.10.filter_layer.filter.w.1 ,
    blocks.10.filter_layer.filter.w.2 ,
    blocks.10.filter_layer.filter.activation.bias ,
    blocks.10.inner_skip.weight ,
    blocks.10.inner_skip.bias ,
    blocks.10.norm1.weight ,
    blocks.10.norm1.bias ,
    blocks.10.mlp.fwd.0.weight ,
    blocks.10.mlp.fwd.0.bias ,
    blocks.10.mlp.fwd.2.weight ,
    blocks.10.mlp.fwd.2.bias ,
    blocks.11.norm0.weight ,
    blocks.11.norm0.bias ,
    blocks.11.filter_layer.filter.wout ,
    blocks.11.filter_layer.filter.w.0 ,
    blocks.11.filter_layer.filter.w.1 ,
    blocks.11.filter_layer.filter.w.2 ,
    blocks.11.filter_layer.filter.activation.bias ,
    blocks.11.norm1.weight ,
    blocks.11.norm1.bias ,
    decoder.fwd.0.weight ,
    decoder.fwd.0.bias ,
    decoder.fwd.2.weight ,
]