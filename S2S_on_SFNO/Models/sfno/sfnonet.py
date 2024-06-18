# (C) Copyright 2023 Nvidia
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from functools import partial
import torch
import torch.nn as nn
import xarray as xr

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


import numpy as np
import os

# from apex.normalization import FusedLayerNorm

from torch.utils.checkpoint import checkpoint

# film_gens
from ..mae.maenet import ContextCast
from ..gcn.gcn import GCN_custom, GCN
from ..vit.vit import ViT



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
        embed_dim_sfno,
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
                embed_dim_sfno,
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
                embed_dim_sfno,
                sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        elif filter_type == "linear" and isinstance(forward_transform, harmonics.RealSHT):
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim_sfno,
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
                embed_dim_sfno,
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
        embed_dim_sfno,
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
        checkpointing_mlp=False,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        # norm layer
        self.norm0 = norm_layer[0]()  # ((h,w))

        # convolution layer
        self.filter_layer = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim_sfno,
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
            self.inner_skip = nn.Conv2d(embed_dim_sfno, embed_dim_sfno, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim_sfno, embed_dim_sfno, 1, bias=False)

        if filter_type == "linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()  # ((h,w))

        if mlp_mode != "none": # default distributed
            mlp_hidden_dim = int(embed_dim_sfno * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim_sfno,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing_mlp=checkpointing_mlp,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim_sfno, embed_dim_sfno, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim_sfno, embed_dim_sfno, 1, bias=False)

    def forward(self, x, *overflow):
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
        embed_dim_sfno,
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
        checkpointing_mlp=False,
    ):
        super().__init__()

        # norm layer
        self.norm0 = norm_layer[0]()  # ((h,w))

        self.film = FiLM()

        # convolution layer
        self.filter_layer = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim_sfno,
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
            self.inner_skip = nn.Conv2d(embed_dim_sfno, embed_dim_sfno, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim_sfno, embed_dim_sfno, 1, bias=False)

        if filter_type == "linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()  # ((h,w))

        if mlp_mode != "none":
            mlp_hidden_dim = int(embed_dim_sfno * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim_sfno,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing_mlp=checkpointing_mlp,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim_sfno, embed_dim_sfno, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim_sfno, embed_dim_sfno, 1, bias=False)

    def global_conv(self,x,residual):
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
        return x


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

        # FiLM
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
        device,
        cfg,
        spectral_transform="sht",
        filter_type="non-linear",
        img_size=(721, 1440),
        scale_factor=6,
        in_chans=73,
        out_chans=73,
        embed_dim_sfno=256,
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
        checkpointing_mlp=False,
        checkpointing_block=False,
        checkpointing_encoder=False,
        checkpointing_decoder=False,
        batch_size = 1,
        **overflow
    ):
        super(FourierNeuralOperatorNet, self).__init__()
        
        self.cfg = cfg
        self.device = device

        self.spectral_transform = spectral_transform
        self.filter_type = filter_type
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim_sfno = self.num_features = embed_dim_sfno
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
        self.checkpointing_mlp = checkpointing_mlp
        self.checkpointing_block = checkpointing_block
        self.checkpointing_encoder = checkpointing_encoder
        self.checkpointing_decoder = checkpointing_decoder
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
                num_features=self.embed_dim_sfno,
                eps=1e-6,
                affine=True,
                track_running_stats=False,
            )
            self.norm_layer1 = self.norm_layer0
        # elif self.normalization_layer == "batch_norm":
        #     norm_layer = partial(nn.InstanceNorm2d, num_features=self.embed_dim_sfno, eps=1e-6, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError(
                f"Error, normalization {self.normalization_layer} not implemented."
            )

        # ENCODER is just an MLP?
        encoder_hidden_dim = self.embed_dim_sfno
        encoder_act = nn.GELU

        # encoder0 = nn.Conv2d(self.in_chans, encoder_hidden_dim, 1, bias=True)
        # encoder1 = nn.Conv2d(encoder_hidden_dim, self.embed_dim_sfno, 1, bias=False)
        # encoder_act = nn.GELU()
        # self.encoder = nn.Sequential(encoder0, encoder_act, encoder1, self.norm_layer0())

        self.encoder = MLP(
            in_features=self.in_chans,
            hidden_features=encoder_hidden_dim,
            out_features=self.embed_dim_sfno,
            output_bias=False,
            act_layer=encoder_act,
            drop_rate=0.0,
            checkpointing_mlp=checkpointing_mlp,
        )

        # self.input_encoding = nn.Conv2d(self.in_chans, self.embed_dim_sfno, 1)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.pos_embed_dim, self.img_size[0], self.img_size[1]))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim_sfno, self.img_size[0], self.img_size[1])
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
                self.embed_dim_sfno,
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
                checkpointing_mlp=self.checkpointing_mlp,
            )

            self.blocks.append(block)

        # DECODER is also an MLP
        decoder_hidden_dim = self.embed_dim_sfno
        decoder_act = nn.GELU

        # decoder0 = nn.Conv2d(self.embed_dim_sfno + self.big_skip*self.in_chans, decoder_hidden_dim, 1, bias=True)
        # decoder1 = nn.Conv2d(decoder_hidden_dim, self.out_chans, 1, bias=False)
        # decoder_act = nn.GELU()
        # self.decoder = nn.Sequential(decoder0, decoder_act, decoder1)

        self.decoder = MLP(
            in_features=self.embed_dim_sfno + self.big_skip * self.in_chans,
            hidden_features=decoder_hidden_dim,
            out_features=self.out_chans,
            output_bias=False,
            act_layer=decoder_act,
            drop_rate=0.0,
            checkpointing_mlp=checkpointing_mlp,
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

        if self.checkpointing_block:
            for blk in self.blocks:
                x = checkpoint(blk,x,use_reentrant=False)
        else:
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


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas,scale=1):
        _gammas = repeat(gammas, 'i j -> i j k l',k=x.shape[2],l=x.shape[3])
        _betas  = repeat(betas, 'i j -> i j k l',k=x.shape[2],l=x.shape[3])
        return ((1+_gammas*scale) * x) + _betas*scale

class FourierNeuralOperatorNet_Filmed(FourierNeuralOperatorNet):
    def __init__(
            self,
            device,
            cfg,
            mlp_ratio=2.0,
            drop_rate=0.0,
            sparsity_threshold=0.0,
            use_complex_kernels=True,
            **kwargs
        ):
        super().__init__(device,cfg,**kwargs)

        # save gamma and beta in model if advanced logging is required
        self.advanced_logging = kwargs["advanced_logging"]
        self.film_layers = kwargs["film_layers"]
        self.depth = kwargs["model_depth"]
        
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
            if self.cfg.repeat_film or i >= self.num_layers - self.film_layers:
                block = FourierNeuralOperatorBlock_Filmed(
                    forward_transform,
                    inverse_transform,
                    self.embed_dim_sfno,
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
                    checkpointing_mlp=self.checkpointing_mlp,
                )
            else:
                block = FourierNeuralOperatorBlock(
                    forward_transform,
                    inverse_transform,
                    self.embed_dim_sfno,
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
                    checkpointing_mlp=self.checkpointing_mlp,
                )
            self.blocks.append(block)
            
        # coarse level =  4 default, could be changed by coarse_level in film_gen arguments
        self.film_gen = Film_wrapper(device,cfg)
        # self.film = FiLM()

    def forward(self, x,sst,scale=1):

        # calculate gammas and betas for film layers
        film_mod = self.film_gen(sst)# None for transformer
        # film_mod = checkpoint(self.film_gen,sst,use_reentrant=False)

        # if film_mod.shape[2] != self.num_layers: 
        #     if self.cfg.repeat_film:
        #         # same film modulation for each block
        #         film_mod = film_mod.expand(-1,-1,self.num_layers,-1)
        #     else:
        #         pass
                # only film modulation on the last #film_layers layers
                # probably unessessary compute ?? for do we still run backward on earlier layers?
                # insead of checking index in block loop just and 0 and let the gamma/beta values unconsidert in normal block. Does is increase mem/compute or does the compute graph realizes that we don't need that then? Err on the save side.
                # shape = list(film_mod.shape)
                # shape[2] = self.num_layers
                # film_mod_temp = torch.zeros(shape).to(self.device)
                # film_mod_temp[:,:,-film_mod.shape[2]:] = film_mod
        gamma,beta = film_mod[:,0],film_mod[:,1] 
        # save gamma and beta in model for validation
        if self.advanced_logging:
            self.gamma = gamma
            self.beta = beta
        
        # save big skip
        if self.big_skip:
            residual = x

        # encoder
        with torch.no_grad():
            if self.checkpointing_encoder:
                x = checkpoint(self.encoder,x,use_reentrant=False)
            else:
                x = self.encoder(x)

            # do positional embedding
            x = x + self.pos_embed

            # forward features
            x = self.pos_drop(x)
        
        if self.checkpointing_block: #self.checkpointing:
            for i, blk in enumerate(self.blocks):
                # if i < 11: # don't want to checkpoint everything? All is needed to be able to go to 4 steps (1|2)
                if self.cfg.repeat_film or i >= self.num_layers - self.film_layers:
                    x = checkpoint(blk,x,gamma[:,i],beta[:,i],scale,use_reentrant=False)
                else:
                    with torch.no_grad():
                        x =checkpoint(blk,x)
        else:
            for i, blk in enumerate(self.blocks):
                if self.cfg.repeat_film or i >= self.num_layers - self.film_layers:
                    film_idx = i - (self.num_layers - self.film_layers)
                    x = blk(x,gamma[:,film_idx],beta[:,film_idx],scale)
                else:
                    with torch.no_grad():
                        x = blk(x)


        # x = self.film(x,gamma[:,0],beta[:,0], 1.0)


        # concatenate the big skip
        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        # decoder
        if self.checkpointing_decoder:
             x = checkpoint(self.decoder,x,use_reentrant=False)
        else:
            x = self.decoder(x)

        return x
    

class Film_wrapper(nn.Module):
    def __init__(self,device,cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg

        num_film_features=256
        self.num_film_features = num_film_features

        if self.cfg.film_gen_type == "gcn":
            self.film_gen = GCN(self.cfg.batch_size,self.device, in_features=self.cfg.temporal_step ,out_features=num_film_features*self.cfg.film_layers*2 , depth=self.cfg.model_depth,embed_dim=self.cfg.embed_dim,assets=os.path.join(self.cfg.assets,"gcn"))# num layers is 1 for now
        elif self.cfg.film_gen_type == "transformer":
            self.film_gen = ViT(patch_size=self.cfg.patch_size[-1], num_classes=num_film_features*self.cfg.film_layers*2, dim=self.cfg.embed_dim, depth=self.cfg.model_depth, heads=16, mlp_dim = self.cfg.mlp_dim, dropout = 0.1, channels = self.cfg.temporal_step, device=self.device)
        elif self.cfg.film_gen_type == "mae":
            if self.cfg.cls is None:
                self.film_gen = ContextCast(self.cfg,data_dim=1,patch_size=self.cfg.patch_size, embed_dim=self.cfg.embed_dim,film_layers=self.cfg.film_layers,)
            # self.film_head = nn.Linear(self.cfg.embed_dim,num_film_features*self.cfg.film_layers*2)
            self.film_head = FeedForward(dim=self.cfg.embed_dim, hidden_dim=self.cfg.mlp_dim, dropout=0.1, out_dim=num_film_features*self.cfg.film_layers*2)
        
            # init
            for x in self.film_head.net: 
                if type(x) == torch.nn.modules.linear.Linear:
                    nn.init.constant_(x.weight, 0)
                    nn.init.constant_(x.bias, 0)
        
        else:
            self.film_gen = GCN_custom(self.cfg.batch_size,self.device,out_features=num_film_features*self.cfg.film_layers*2, depth=self.cfg.model_depth,embed_dim=self.cfg.embed_dim,assets=os.path.join(self.cfg.assets,"gcn"))# num layers is 1 for now
    
    def get_parameters(self):
        if self.cfg.film_gen_type == "mae":
            return self.film_head.parameters()
        return self.film_gen.parameters()

    def forward(self,sst):
       # Mae
        if self.cfg.film_gen_type == "mae":
            if self.cfg.cls is None:
                pass
                # with torch.no_grad():
                    # cls = self.film_gen(sst)[-1]
                # x = self.film_head(cls)
            else:
                x = self.film_head(sst)
        else:
            x = self.film_gen(sst)
        return x.reshape(sst.shape[0],2,self.cfg.film_layers,self.num_film_features)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.,out_dim=256):
        super().__init__()
        self.out_features = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
        