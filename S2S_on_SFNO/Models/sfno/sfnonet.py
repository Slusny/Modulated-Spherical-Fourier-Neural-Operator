# (C) Copyright 2023 Nvidia
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool

import numpy as np
import os
import einops

# from apex.normalization import FusedLayerNorm

from torch.utils.checkpoint import checkpoint


# helpers
from .layers import (
    trunc_normal_,
    DropPath,
    MLP,
)
from .layers import (
    SpectralAttentionS2,
    SpectralConvS2,
)
from .layers import (
    SpectralAttention2d,
    SpectralConv2d,
)

from .layers import GraphConvolution

import torch_harmonics as harmonics

# to fake the sht module with ffts
from .layers import RealFFT2, InverseRealFFT2

from .contractions import *

# from .fourcastnetv2 import activations
from .activations import *


class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        hidden_size_factor=2,
        compression=None,
        rank=128,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        drop_rate=0.0,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and isinstance(
            forward_transform, harmonics.RealSHT
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold,
                use_complex_network=complex_network,
                use_complex_kernels=use_complex_kernels,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )
        elif filter_type == "non-linear" and isinstance(forward_transform, RealFFT2):
            self.filter = SpectralAttention2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        elif filter_type == "linear" and isinstance(forward_transform, RealSHT):
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                compression=compression,
                rank=rank,
                bias=False,
            )

        elif filter_type == "linear" and isinstance(forward_transform, RealFFT2):
            self.filter = SpectralConv2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                compression=compression,
                rank=rank,
                bias=False,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        # num_blocks = 8,
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        compression=None,
        rank=128,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        mlp_mode="none",
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=False,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        # norm layer
        self.norm0 = norm_layer[0]()  # ((h,w))

        # convolution layer
        self.filter_layer = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            compression=compression,
            rank=rank,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate,
        )

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()  # ((h,w))

        if mlp_mode != "none":
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def forward(self, x):
        residual = x

        x = self.norm0(x)
        x = self.filter_layer(x).contiguous()

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x = self.norm1(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x


class FourierNeuralOperatorBlock_Filmed(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        block_idx=0,
        act_layer=nn.GELU,
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        # num_blocks = 8,
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        compression=None,
        rank=128,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        mlp_mode="none",
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=False,
    ):
        super().__init__()

        # norm layer
        self.norm0 = norm_layer[0]()  # ((h,w))

        self.film = FiLM()

        # convolution layer
        self.filter_layer = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            compression=compression,
            rank=rank,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate,
        )

        self.block_idx = block_idx

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()  # ((h,w))

        if mlp_mode != "none":
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def forward(self, x, gamma, beta, scale=1):
        residual = x
        
        x = self.norm0(x)
        x = self.filter_layer(x).contiguous()

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x = self.norm1(x)

        x = self.film(x,gamma,beta, scale)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x

    # @torch.jit.ignore
    # def checkpoint_forward(self, x):
    #     return checkpoint(self._forward, x)

    # def forward(self, x):
    #     if self.checkpointing:
    #         return self.checkpoint_forward(x)
    #     else:
    #         return self._forward(x)


class FourierNeuralOperatorNet(nn.Module):
    def __init__(
        self,
        spectral_transform="sht",
        filter_type="non-linear",
        img_size=(721, 1440),
        scale_factor=6,
        in_chans=73,
        out_chans=73,
        embed_dim=256,
        num_layers=12,
        mlp_mode="distributed",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.0,
        normalization_layer="instance_norm",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=True,
        compression=None,
        rank=128,
        complex_network=True,
        complex_activation="real",
        spectral_layers=3,
        laplace_weighting=False,
        checkpointing=False,
        batch_size = 1,
        **overflow
    ):
        super(FourierNeuralOperatorNet, self).__init__()

        self.spectral_transform = spectral_transform
        self.filter_type = filter_type
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = self.num_features = embed_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.mlp_mode = mlp_mode
        self.big_skip = big_skip
        self.compression = compression
        self.rank = rank
        self.complex_network = complex_network
        self.complex_activation = complex_activation
        self.spectral_layers = spectral_layers
        self.laplace_weighting = laplace_weighting
        self.checkpointing = checkpointing
        self.batch_size = batch_size
        

        # compute downsampled image size
        self.h = self.img_size[0] // self.scale_factor
        self.w = self.img_size[1] // self.scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)] # made class property from original local variable

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            self.norm_layer0 = partial( #changed norm_layer0 to self.norm_layer0
                nn.LayerNorm,
                normalized_shape=(self.img_size[0], self.img_size[1]),
                eps=1e-6,
            )
            self.norm_layer1 = partial(
                nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6
            )
        elif self.normalization_layer == "instance_norm":
            self.norm_layer0 = partial(
                nn.InstanceNorm2d,
                num_features=self.embed_dim,
                eps=1e-6,
                affine=True,
                track_running_stats=False,
            )
            self.norm_layer1 = self.norm_layer0
        # elif self.normalization_layer == "batch_norm":
        #     norm_layer = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError(
                f"Error, normalization {self.normalization_layer} not implemented."
            )

        # ENCODER is just an MLP?
        encoder_hidden_dim = self.embed_dim
        encoder_act = nn.GELU

        # encoder0 = nn.Conv2d(self.in_chans, encoder_hidden_dim, 1, bias=True)
        # encoder1 = nn.Conv2d(encoder_hidden_dim, self.embed_dim, 1, bias=False)
        # encoder_act = nn.GELU()
        # self.encoder = nn.Sequential(encoder0, encoder_act, encoder1, self.norm_layer0())

        self.encoder = MLP(
            in_features=self.in_chans,
            hidden_features=encoder_hidden_dim,
            out_features=self.embed_dim,
            output_bias=False,
            act_layer=encoder_act,
            drop_rate=0.0,
            checkpointing=checkpointing,
        )

        # self.input_encoding = nn.Conv2d(self.in_chans, self.embed_dim, 1)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.pos_embed_dim, self.img_size[0], self.img_size[1]))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1])
        )

        # prepare the SHT
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

        if self.spectral_transform == "sht":
            self.trans_down = harmonics.RealSHT(
                *self.img_size, lmax=modes_lat, mmax=modes_lon, grid="equiangular"
            ).float()
            self.itrans_up = harmonics.InverseRealSHT(
                *self.img_size, lmax=modes_lat, mmax=modes_lon, grid="equiangular"
            ).float()
            self.trans = harmonics.RealSHT(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()
            self.itrans = harmonics.InverseRealSHT(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()

            # we introduce some ad-hoc rescaling of the weights to aid gradient computation:
            sht_rescaling_factor = 1e5
            self.trans_down.weights = self.trans_down.weights * sht_rescaling_factor
            self.itrans_up.pct = self.itrans_up.pct / sht_rescaling_factor
            self.trans.weights = self.trans.weights * sht_rescaling_factor
            self.itrans.pct = self.itrans.pct / sht_rescaling_factor

        elif self.spectral_transform == "fft":
            self.trans_down = RealFFT2(
                *self.img_size, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = InverseRealFFT2(
                *self.img_size, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.trans = RealFFT2(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans = InverseRealFFT2(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear" if 0 < i < self.num_layers - 1 else None
            outer_skip = "identity" if 0 < i < self.num_layers - 1 else None
            mlp_mode = self.mlp_mode if not last_layer else "none"

            if first_layer:
                norm_layer = (self.norm_layer0, self.norm_layer1)
            elif last_layer:
                norm_layer = (self.norm_layer1, self.norm_layer0)
            else:
                norm_layer = (self.norm_layer1, self.norm_layer1)

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=self.filter_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=self.dpr[i],
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                mlp_mode=mlp_mode,
                compression=self.compression,
                rank=self.rank,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
            )

            self.blocks.append(block)

        # DECODER is also an MLP
        decoder_hidden_dim = self.embed_dim
        decoder_act = nn.GELU

        # decoder0 = nn.Conv2d(self.embed_dim + self.big_skip*self.in_chans, decoder_hidden_dim, 1, bias=True)
        # decoder1 = nn.Conv2d(decoder_hidden_dim, self.out_chans, 1, bias=False)
        # decoder_act = nn.GELU()
        # self.decoder = nn.Sequential(decoder0, decoder_act, decoder1)

        self.decoder = MLP(
            in_features=self.embed_dim + self.big_skip * self.in_chans,
            hidden_features=decoder_hidden_dim,
            out_features=self.out_chans,
            output_bias=False,
            act_layer=decoder_act,
            drop_rate=0.0,
            checkpointing=checkpointing,
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):  # or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        # save big skip
        if self.big_skip:
            residual = x

        # encoder
        x = self.encoder(x)

        # do positional embedding
        x = x + self.pos_embed

        # forward features
        x = self.forward_features(x)

        # concatenate the big skip
        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        # decoder
        x = self.decoder(x)

        return x

class GCN(torch.nn.Module):
    def __init__(self,batch_size,device,out_features=256,num_layers=12,coarse_level=4,graph_asset_path="/mnt/qb/work2/goswami0/gkd965/Assets/gcn"):
        super().__init__()

        # Model
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = out_features*2
        self.activation = nn.LeakyReLU()
        self.conv1 = GCNConv(1, self.hidden_size,cached=True)
        self.perceptive_field = 3
        self.conv_layers = [GCNConv(self.hidden_size, self.hidden_size,cached=True) for _ in range(self.perceptive_field)]
        self.heads_gamma = nn.ModuleList([])
        self.heads_beta = nn.ModuleList([])
        for i in range(self.num_layers):
            self.heads_gamma.append(nn.Linear(self.hidden_size, out_features))
            self.heads_beta.append(nn.Linear(self.hidden_size, out_features))
        
        # prepare the graph 

        # Nan_mask removes nans from sst-matrix -> 1D array, edge_index and nan_mask loaded from file    
        # !! if numpy masking is used to downsample instead of coarsen, an other edge_index, nan_mask needs to be loaded
        edge_index = torch.load(os.path.join(graph_asset_path,"edge_index_coarsen_"+str(coarse_level)+".pt"))
        nan_mask = np.load(os.path.join(graph_asset_path,"nan_mask_coarsen_"+str(coarse_level)+"_notflatten.npy"))
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
        # # x = einops.repeat(x,'i j -> i (repeat j)',repeat=self.hidden_size) + F.leaky_relu(x1)
        # x = x + F.leaky_relu(x1)
        # # x = F.dropout(x, training=self.training)
        # x2 = self.conv2(x, self.edge_index_batch)
        # x = x + F.leaky_relu(x2)
        # x3 = self.conv2(x, self.edge_index_batch)
        # x = x + F.leaky_relu(x3)
        # # x = x + self.conv2(x, self.edge_index_batch)
        # x =  self.conv2(x, self.edge_index_batch)
        # x = global_mean_pool(x, self.batch)
        heads_gamma = []
        heads_beta = []
        for i in range(self.num_layers):
            heads_gamma.append(self.heads_gamma[i](x))
            heads_beta.append(self.heads_beta[i](x))
        return torch.stack([torch.stack(heads_gamma),torch.stack(heads_beta)]).squeeze() # shape: (2,num_layers,batch_size,embed_dim)



class GCN_custom(nn.Module):
    def __init__(self,batch_size,device,out_features=256,num_layers=12,coarse_level=4,graph_asset_path="/mnt/qb/work2/goswami0/gkd965/Assets/gcn"):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = out_features*2
        self.conv1 = GraphConvolution(1, self.hidden_size)
        self.perceptive_field = 6 # 3
        self.conv_layers = nn.ModuleList([GraphConvolution(self.hidden_size, self.hidden_size) for _ in range(self.perceptive_field)])
        self.activation = nn.LeakyReLU() # change parameter for leaky relu also in initalization of GraphConvolution layer
        self.heads_gamma = nn.ModuleList([])
        self.heads_beta = nn.ModuleList([])
        for i in range(self.num_layers):
            self.heads_gamma.append(nn.Linear(self.hidden_size, out_features))
            self.heads_beta.append(nn.Linear(self.hidden_size, out_features))

        ## Prepare Graph
        # load sparse adjacentcy matrix from file ( shape: num_node x num_nodes )
        # self.adj = torch.load(os.path.join(graph_asset_path,"adj_coarsen_"+str(coarse_level)+"_sparse.pt")).to(device)
        # dense 
        self.adj = torch.load(os.path.join(graph_asset_path,"adj_coarsen_"+str(coarse_level)+"_fully.pt")).to(device)
        
        # nan mask masks out every nan value on land ( shape: 180x360 for 1° data )
        self.nan_mask = np.load(os.path.join(graph_asset_path,"nan_mask_coarsen_"+str(coarse_level)+"_notflatten.npy"))
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
        heads_gamma = []
        heads_beta = []
        for i in range(self.num_layers):
            heads_gamma.append(self.heads_gamma[i](x))
            heads_beta.append(self.heads_beta[i](x))
        return torch.stack([torch.stack(heads_gamma),torch.stack(heads_beta)]).squeeze() # shape: (2,num_layers,batch_size,embed_dim)


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas,scale=1):
        _gammas = einops.repeat(gammas, 'j -> i j k l',i=x.shape[0],k=x.shape[2],l=x.shape[3])
        _betas  = einops.repeat(betas, 'j -> i j k l',i=x.shape[0],k=x.shape[2],l=x.shape[3])
        return ((1+_gammas*scale) * x) + _betas*scale

class FourierNeuralOperatorNet_Filmed(FourierNeuralOperatorNet):
    def __init__(
            self,
            device,
            mlp_ratio=2.0,
            drop_rate=0.0,
            sparsity_threshold=0.0,
            use_complex_kernels=True,
            **kwargs
        ):
        super().__init__(**kwargs)

        # save gamma and beta in model if advanced logging is required
        self.advanced_logging = kwargs["advanced_logging"]
        
        # new SFNO-Block with Film Layer
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear" if 0 < i < self.num_layers - 1 else None
            outer_skip = "identity" if 0 < i < self.num_layers - 1 else None
            mlp_mode = self.mlp_mode if not last_layer else "none"

            if first_layer:
                norm_layer = (self.norm_layer0, self.norm_layer1)
            elif last_layer:
                norm_layer = (self.norm_layer1, self.norm_layer0)
            else:
                norm_layer = (self.norm_layer1, self.norm_layer1)

            block = FourierNeuralOperatorBlock_Filmed(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=self.filter_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=self.dpr[i],
                block_idx = i,
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                mlp_mode=mlp_mode,
                compression=self.compression,
                rank=self.rank,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
            )

            self.blocks.append(block)
        
        if kwargs["film_gen_type"] == "gcn":
            self.film_gen = GCN(self.batch_size,device,out_features=self.embed_dim,num_layers=1)# num layers is 1 for now
        elif kwargs["film_gen_type"] == "transformer":
            pass
        else:
            self.film_gen = GCN_custom(self.batch_size,device,out_features=self.embed_dim,num_layers=1)# num layers is 1 for now
    
    def forward(self, x,sst,scale=1):

        # calculate gammas and betas for film layers
        gamma,beta = self.film_gen(sst)
        # save gamma and beta in model for validation
        if self.advanced_logging:
            self.gamma = gamma
            self.beta = beta
        
        if gamma.shape[0] != self.num_layers:
            gamma = gamma.repeat(self.num_layers,1)
            beta = beta.repeat(self.num_layers,1)
        
        # save big skip
        if self.big_skip:
            residual = x

        # encoder
        x = self.encoder(x)

        # do positional embedding
        x = x + self.pos_embed

        # forward features
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x,gamma[i],beta[i],scale)

        # concatenate the big skip
        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        # decoder
        x = self.decoder(x)

        return x
    
    # get only the parameters of the film generator
    def get_film_params(self):
        return self.film_gen.parameters()
