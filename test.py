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
# # t.mod()
# # t.p()

# import torch
# from time import sleep
# import numpy as np
# import traceback
# # class net(torch.nn.Module):
# #     def __init__(self):
# #         super(net,self).__init__()
# #         self.lin = torch.nn.Linear(3,3)
    
# #     def forward(self,x):
# #         return self.lin(x)
    
# # a = net()
# # x = torch.tensor([1,np.nan,3])
# # o = a(x)
# # # print(o)


# # k="hi "
# # try:
# #     for i in range(100000):
# #         print(i)
# #         k += str(i)
# #         sleep(2)
# # except Exception as e:
# #     print(e)
# # except KeyboardInterrupt as kk:
# #     print(kk)
# #     print(traceback.format_exc())
# #     print("done1")
# # except :
# #     print(k)
# #     print("done")

# # print("haa")



# for i in range(5):
#     print(i)
#     i = -1

from main import return_trainer
import torch
import matplotlib.pyplot as plt

trainer = return_trainer([
    '--model','mae',
    '--assets','/mnt/ssd2/Master/S2S_on_SFNO/Assets',
    '--trainingdata-path',"/mnt/V/wb2_2001-2003.zarr",
    '--resume-checkpoint',"/mnt/V/Master/checkpoints/summer-puddle-7/checkpoint_mae_latest_None_iter=423_epoch=2.pkl",
    '--trainingset-start-year','2001',
    '--trainingset-end-year','2002',
    '--validationset-start-year','2002',
    '--validationset-end-year','2003',
    '--loss-fn','NormalCRPS',
    '--loss-reduction','mean',
    '--nan-mask-threshold','0.8',
    ])

checkpoint_list = ["/mnt/V/Master/checkpoints/summer-puddle-7/checkpoint_mae_latest_None_iter=423_epoch=2.pkl","save-path","/mnt/V/Master/checkpoints"]

trainer.set_dataloader()
trainer.util.load_statistics()
trainer.util.set_seed(42) 
trainer.create_loss() 


with torch.no_grad():
    for cp_idx, checkpoint in enumerate(checkpoint_list):
        cp = trainer.util.load_model(checkpoint)
        # trainer.save_path = save_path
        for i, data in enumerate(trainer.validation_loader):
            for step in range(trainer.cfg.multi_step_validation+1):
                data_nn = data[step][0].to(trainer.util.device)
                data_norm = trainer.util.normalise(data[step][0]).to(trainer.util.device)
                outputs, gt = trainer.model_forward(data_norm,data_norm,0)
                break
            break
        break

loss = trainer.get_loss(outputs,gt)

plt.figure()
plt.imshow(loss.cpu().numpy()[0,0,0])
plt.colorbar()
plt.show()