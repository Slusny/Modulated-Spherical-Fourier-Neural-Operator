
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import os
import numpy as np

from .layers import GraphConvolution

class GCN(torch.nn.Module):
    def __init__(self,batch_size,device,depth=8,embed_dim=512, out_features=256,coarse_level=4,assets="/mnt/qb/work2/goswami0/gkd965/Assets/gcn"):
        super().__init__()

        # Model
        self.batch_size = batch_size
        self.hidden_size = embed_dim
        self.out_features = out_features
        self.activation = nn.LeakyReLU()
        self.conv1 = GCNConv(1, self.hidden_size,cached=True)
        self.perceptive_field = depth
        self.conv_layers = nn.ModuleList([GCNConv(self.hidden_size, self.hidden_size,cached=True) for _ in range(self.perceptive_field)])
        self.head_film = nn.Linear(self.hidden_size,out_features )

        # with torch.no_grad():
        self.head_film.weight = nn.Parameter(torch.zeros_like(self.head_film.weight))
        self.head_film.bias = nn.Parameter(torch.zeros_like(self.head_film.bias))
        
        # prepare the graph 
        # Nan_mask removes nans from sst-matrix -> 1D array, edge_index and nan_mask loaded from file    
        # !! if numpy masking is used to downsample instead of coarsen, an other edge_index, nan_mask needs to be loaded
        edge_index = torch.load(os.path.join(assets,"edge_index_coarsen_"+str(coarse_level)+".pt"))
        nan_mask = np.load(os.path.join(assets,"nan_mask_coarsen_"+str(coarse_level)+"_notflatten.npy"))
        num_node_features = 686364
        num_nodes = np.sum(nan_mask)
        num_edges = edge_index.shape[1]

        # repeat nan_mask for each sst-matrix in batch 
        self.batch_nan_mask = np.repeat(nan_mask[ np.newaxis,: ], batch_size, axis=0)

        # handle batch by appending sst-matrices to long 1D array, edge_index gets repeated and offseted to create the distinct graphs 
        self.batch = torch.tensor(list(range(batch_size))*num_nodes).reshape((num_nodes,batch_size)).T.flatten().to(device)
        offset_ = torch.tensor(list(range(batch_size))*num_edges).reshape((num_edges,batch_size)).T.flatten()*num_nodes
        offset = torch.stack([offset_,offset_])
        self.edge_index_batch = ( edge_index.repeat((1,batch_size))+offset ).to(device)

        # shape anlysis
        # node values (sst): (num_nodes,1) ...

    def forward(self, sst):
        x = sst[self.batch_nan_mask][None].T
        # orig = x
        # # No Skip
        # h = self.conv1(x, self.edge_index_batch)
        # h = F.relu(h)
        # # x = F.dropout(x, training=self.training)
        # h = self.conv2(h, self.edge_index_batch)
        # h = F.relu(h)
        # h = self.conv2(h, self.edge_index_batch)
        # h = F.relu(h)
        # h = self.conv2(h, self.edge_index_batch)
        # h = global_mean_pool(h, self.batch)

        # # No Skip
        # x = self.activation(self.conv1(x, self.edge_index_batch))
        # for conv in self.conv_layers:
        #     x =  self.activation(conv(x, self.edge_index_batch))
        # x = global_mean_pool(x, self.batch)
        

        # Skip
        x = x + self.activation(self.conv1(x, self.edge_index_batch))
        for conv in self.conv_layers:
            x = x + self.activation(conv(x, self.edge_index_batch))
        x = global_mean_pool(x, self.batch)

        # # Skip
        # x1 = self.conv1(x, self.edge_index_batch)
        # # x = repeat(x,'i j -> i (repeat j)',repeat=self.hidden_size) + F.leaky_relu(x1)
        # x = x + F.leaky_relu(x1)
        # # x = F.dropout(x, training=self.training)
        # x2 = self.conv2(x, self.edge_index_batch)
        # x = x + F.leaky_relu(x2)
        # x3 = self.conv2(x, self.edge_index_batch)
        # x = x + F.leaky_relu(x3)
        # # x = x + self.conv2(x, self.edge_index_batch)
        # x =  self.conv2(x, self.edge_index_batch)
        # x = global_mean_pool(x, self.batch)
        return self.head_film(x)#.squeeze()
        


# Blowes up STD (7 layers -> std=8.3)
class GCN_custom(nn.Module):
    def __init__(self,batch_size,device,depth,embed_dim=512, out_features=256,coarse_level=4,assets="/mnt/qb/work2/goswami0/gkd965/Assets/gcn"):
        """
        Paramters: last lin layer: 131072, conv hidden layer (sparse): 262144
        But Pararmeters SFNO: 
            blocks.7.norm0.weight                   256
            blocks.7.norm0.bias                     256
            blocks.7.filter_layer.filter.wout       262144
            blocks.7.filter_layer.filter.w.0        262144
            blocks.7.filter_layer.filter.w.1        524288
            blocks.7.filter_layer.filter.w.2        524288
            blocks.7.inner_skip.weight              65536
            blocks.7.inner_skip.bias                256
            blocks.7.norm1.weight                   256
            blocks.7.norm1.bias                     256
            blocks.7.mlp.fwd.0.weight               131072
            blocks.7.mlp.fwd.0.bias                 512
            blocks.7.mlp.fwd.2.weight               131072
            blocks.7.mlp.fwd.2.bias                 256
            pos_embed        265789440 ??
        """
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = embed_dim
        self.out_features = out_features
        self.conv1 = GraphConvolution(1, self.hidden_size)
        self.perceptive_field = depth # 3
        self.conv_layers = nn.ModuleList([GraphConvolution(self.hidden_size, self.hidden_size) for _ in range(self.perceptive_field)])
        self.activation = nn.LeakyReLU() # change parameter for leaky relu also in initalization of GraphConvolution layer
        self.head_film = nn.Linear(self.hidden_size, out_features)

        # Set film weights to 0
        # with torch.no_grad():
        self.head_film.weight = nn.Parameter(torch.ones_like(self.head_film.weight))
        self.head_film.bias = nn.Parameter(torch.zeros_like(self.head_film.bias))
            # self.head_film.weight[0, 0] = 2.
            # model[0].weight.fill_(3.)

        ## Prepare Graph
        # load sparse adjacentcy matrix from file ( shape: num_node x num_nodes )
        self.adj = torch.load(os.path.join(assets,"adj_coarsen_"+str(coarse_level)+"_sparse.pt")).to(device)
        # sparse adj needs 0.01  GB on memory
        # dense 7
        
        # self.adj = torch.load(os.path.join(graph_asset_path,"adj_coarsen_"+str(coarse_level)+"_fully.pt")).to(device)
        # # adj matrix takes 8.16 GB on memory
        
        # nan mask masks out every nan value on land ( shape: 180x360 for 1° data )
        self.nan_mask = np.load(os.path.join(assets,"nan_mask_coarsen_"+str(coarse_level)+"_notflatten.npy"))
        # repeat nan mask in batch size dimension ( shape: batch_sizex180x360 )
        self.batch_nan_mask = np.repeat(self.nan_mask[ np.newaxis,: ], batch_size, axis=0)
        
    def forward(self, sst):
        # x.shape: batch_size x num_nodes x features
        x = sst[self.batch_nan_mask].reshape(self.batch_size,-1,1)
        
        # No Skip
        # x = self.activation(self.conv1(x, self.adj))
        # for conv in self.conv_layers:
        #     x = self.activation(conv(x, self.adj))
        # x = x.mean(dim=-2)

        # Skip
        x = x + self.activation(self.conv1(x, self.adj))
        for conv in self.conv_layers:
            x = x + self.activation(conv(x, self.adj))
        x = x.mean(dim=-2)
    
        # Film Heads
        return self.head_film(x)#.squeeze()
    
