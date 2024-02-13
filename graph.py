# %%
import xarray as xr
import os
import numpy as np
import sys
import json
from datetime import datetime
from S2S_on_SFNO.Models.provenance import system_monitor
from multiprocessing import Pool, active_children
from time import sleep



# %%
from time import time

# %%

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)
# %%
class GCN(torch.nn.Module):
    def __init__(self,out_features=256,num_layers=12):
        super().__init__()
        self.num_layers = num_layers
        num_node_features = 686364
        hidden_size = out_features*2
        self.conv1 = GCNConv(1, hidden_size,cached=True)
        self.conv2 = GCNConv(hidden_size, hidden_size,cached=True)
        self.fc1 = torch.nn.Linear(hidden_size, out_features)
        self.heads_gamma = nn.ModuleList([])
        self.heads_beta = nn.ModuleList([])
        for i in range(self.num_layers):
            self.heads_gamma.append(nn.Linear(hidden_size, out_features))
            self.heads_beta.append(nn.Linear(hidden_size, out_features))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # batch = torch.tensor([0]*x.size().numel(),dtype=torch.long)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        heads_gamma = []
        heads_beta = []
        for i in range(self.num_layers):
            heads_gamma.append(self.heads_gamma[i](x))
            heads_beta.append(self.heads_beta[i](x))
        return torch.stack([torch.stack(heads_gamma),torch.stack(heads_beta)]).squeeze()


# %%
path = "/mnt/V/Master/data/"
# path = "/mnt/qb/goswami/data/era5/single_pressure_level/"
ds = xr.open_dataset(path + "sea_surface_temperature/sea_surface_temperature_2019.nc")
sst = ds.sel(time="2019-01-01T00:00").to_array()[0].to_numpy()

# # %%
# edge_index = torch.load("/mnt/V/Master/model/edge_index.pt")
# nan_mask = np.load("/mnt/V/Master/model/nan_mask.npy")

edge_index = torch.load("/mnt/V/Master/model/edge_index_coarsen_4.pt")
nan_mask = np.load("/mnt/V/Master/model/nan_mask_coarsen_4.npy")


# %%
sst_5 = ds.sel(time=slice("2019-01-01T00:00","2019-01-01T00:00")).coarsen(latitude=4,longitude=4,boundary='trim').mean() #.coarsen(latitude=7,longitude=8)
sst_5 = sst_5.to_array()[0].to_numpy()
batch_size = sst_5.shape[0]
print(sst_5.shape)


# %%
num_nodes = np.sum(nan_mask)
num_nodes

# %%
num_edges = edge_index.shape[1]
num_edges

# %%
# repeat mask along batch
batch_nan_mask = np.repeat(nan_mask[ np.newaxis,: ], batch_size, axis=0)
# flatten lat long
sst_5_rs = sst_5.reshape(batch_size,-1)


# %%


# %%



# %%
# along same dim
sst_5_nn = sst_5_rs[batch_nan_mask]
print(sst_5_nn.shape)

batch = torch.tensor(list(range(batch_size))*num_nodes).reshape((num_nodes,batch_size)).T.flatten()
offset_ = torch.tensor(list(range(batch_size))*num_edges).reshape((num_edges,batch_size)).T.flatten()*num_nodes
offset = torch.stack([offset_,offset_])

edge_index_batch = edge_index.repeat((1,batch_size))+offset

# %%
data_batch = Data(x=torch.tensor(sst_5_nn)[None].T, edge_index=edge_index_batch,batch=batch)

# %%
offset[:,0]


# %%
edge_index.repeat((1,batch_size))[:,0]

# %%
edge_index_batch.dtype

# %%
data_batch.x.dtype

# %%
# explicit batch dim

# sst_5_nn = sst_5_rs[batch_nan_mask]
# print(sst_5_nn.shape)
# set_5_nnn = sst_5_nn.reshape((batch_size,num_nodes))

# batch = np.array(list(range(batch_size))*num_nodes).reshape((num_nodes,batch_size)).T
# data = Data(x=torch.tensor(set_5_nnn), edge_index=edge_index,batch=batch.flatten())

# %%
# def kernel(i,j,w,h):
#     x_temp = i+w
#     y_temp = j+h

#     if w == 0:
#         x = x_temp
#     elif x_temp < 0:
#         x = lat + x_temp
#     elif x_temp >= lat:
#         x = x_temp - lat
#     else:
#         x = x_temp
    

#     if h == 0:
#         y = y_temp
#     elif y_temp < 0:
#         y = long + y_temp
#     elif y_temp >= long:
#         y = y_temp - long
#     else:
#         y = y_temp
#     return (x,y)

# %% [markdown]
# following file:///home/lenny/Downloads/MGCN_SST_R2_v01.pdf

# # %%
# point = []
# neigh = []
# #shape: (num_nodes,features)
# lat,long = sst.shape
# idx = 0
# skip_point = False
# for i in range(lat):
#     for j in range(long):
#         for w in [-1,0,1]:
#             for h in [-1,0,1]:
#                 if np.isnan(sst[i,j]):
#                     skip_point = True
#                     continue
#                 if w==0 and h==0:
#                     continue
#                 neighbour = kernel(i,j,w,h)
#                 if np.isnan(sst[neighbour]):
#                     continue
#                 point.append(idx)
#                 neigh.append(neighbour[0]*lat+neighbour[1])
#         if not skip_point:
#             idx += 1
#         else:
#             skip_point = False
# print(idx)
# edge_index = torch.tensor([point,neigh],dtype=torch.long)

# # %%
# sst_f = sst.flatten()
# sst_nn = sst_f[~np.isnan(sst_f)][None].T

# %%
# torch.save(edge_index,"/mnt/V/Master/model/edge_index.pt")
# np.save("/mnt/V/Master/model/nan_mask",~np.isnan(sst_f))

# %%


# %%


# %%


# %%





# %%
# data = Data(x=torch.tensor(sst_nn)[None], edge_index=edge_index,batch=[0]*sst_nn.size)

# %%


# %%
g = GCN()
g.to(device)
# %%
start = time()
out = g(data_batch.to(device))
end = time()
print(end-start)

# %%
print(g)

# # %%
# ds_msp = xr.open_dataset("/mnt/qb/goswami/data/era5/single_pressure_level/mean_sea_level_pressure/mean_sea_level_pressure_2019.nc")
# sl_msp = ds_msp.sel(time="2019-01-01T00:00").to_array()

# # %%


# # %%
# import matplotlib.pyplot as plt

# # %%

# sl.to_array().plot()

# # %%

# sl_msp.plot()

# # %%
# sst_np = sl.to_numpy()

# # %%
# sst_t = np.invert(np.isnan(sst_np)[0])

# # %%
# plt.imshow(sst_t)

# # %%
# num_sst_points = np.sum(sst_t)
# print("number of sst points: ",num_sst_points)

# # %%
# total_points = sst_t.shape[0]*sst_t.shape[1]

# # %%
# percent_water = num_sst_points/total_points
# print("number of Water Points: ",round(percent_water,2)*100,"%")

# # %%
# embed_dim = 256
# scaling = 100

# # %%
# for i in range(1,8):
#     print(num_sst_points//(scaling*i))

# # %%
# num_sst_points//16

# # %%
# num_sst_points//(16*4)

# # %%



