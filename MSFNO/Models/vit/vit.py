import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1) 
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # my first idea
        # attn = self.attend(torch.nan_to_num(dots,nan=-torch.inf))
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# This torch module realises patch embedding for the transformer and handles nan values in the input
# class Transformer_patch_embedding(nn.Module):
#     def __init__(self, patch_height, patch_width, patch_dim, dim):
#         super().__init__()
        
#         self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
#         self.norm1 = nn.LayerNorm(patch_dim)
#         self.lin = nn.Linear(patch_dim, dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.mask = None

#     def rm_embed_nan(self, x, batch):
#         if not torch.is_tensor(self.mask):
#             self.mask = torch.any(torch.isnan(x),dim=-1).logical_not()[0] # keeps batch dimension in mask and removes it by this # x is rearranged sst to patches 
#         return x[...,self.mask,:]


#     def forward(self, x):
#         batch = x.shape[0] ## not correct
#         x = self.rearrange(x)
#         x = self.rm_embed_nan(x,batch)
#         x = self.norm1(x)
#         x = self.lin(x)
#         x = self.norm2(x)
#         return x

class Transformer_patch_embedding(nn.Module):
    '''
    Transforms the input tensor to patches and removes nan values
    If a patch has a certain ratio of nan values (nan_threshold), the whole patch is removed
    The nan_mask_th removes all the tokens that go above the threshold
    The nan_mask keeps track at which position in the token an nan value was present
    When the token that had only a few nan values gets reconstruced, these positions are filled again with nan values
    '''
    def __init__(self, patch_time, patch_height, patch_width, patch_dim, dim, nan_mask_threshold):
        super().__init__()
        
        self.rearrange = Rearrange('b (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw)',pt = patch_time, ph = patch_height, pw = patch_width)  
        self.norm1 = nn.LayerNorm(patch_dim)
        self.lin = nn.Linear(patch_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.nan_mask_threshold = nan_mask_threshold
        self.nan_mask_th = None

    def rm_embed_nan(self, x):
        # x are the rearranged sst to patches
        # compute masks only onece for dataset
        if not torch.is_tensor(self.nan_mask_th):
            nan_threshold = 0.5
            self.nan_mask = torch.isnan(x[0])
            nan_ratio_per_token = self.nan_mask.sum(dim=-1)/x.shape[-1]
            self.nan_mask_th = nan_ratio_per_token < self.nan_mask_threshold 

        x = x[...,self.nan_mask_th,:]
        # fill the remaining nan values with 0 to be useable in the trasformer
        x = torch.nan_to_num(x, nan=0.0)
        return x

    def forward(self, x):
        if len(x.shape) == 3:
            # add data dimension if not present (features in data)
            x.unsqueeze_(1)
        x = self.rearrange(x)
        x = self.rm_embed_nan(x)
        x = self.norm1(x)
        x = self.lin(x)
        x = self.norm2(x)
        return x, self.nan_mask, self.nan_mask_th


class ViT(nn.Module):
    '''
    Doesn't look at temporal dim
    '''
    # simple vit doesn't have dropout, different pos emb and no cls token
    def __init__(self, *, patch_size, num_classes, dim, depth, heads, mlp_dim, nan_mask_threshold=0.5, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., coarse_level=4,device="cpu"):
        super().__init__()
        image_height, image_width = 721//coarse_level, 1440//coarse_level #pair(image_size)
        patch_time, patch_height, patch_width = patch_size
        self.device = device
        self.dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (28//patch_time)
        patch_dim = patch_height * patch_width * patch_time
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.to_patch_embedding = Transformer_patch_embedding(*patch_size, patch_dim, dim, nan_mask_threshold)

        # if we load the mask from  file once instead of calculating it each time in the forward pass, do we save significant run time?
        # self.nan_mask = np.load(os.path.join(graph_asset_path,"nan_mask_coarsen_"+str(coarse_level)+"_notflatten.npy"))
        self.nan_mask = None

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )
        self.encoder_position_code = torch.nn.init.normal_(nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad = True), std = 0.2)
        self.decoder_position_code = torch.nn.init.normal_(nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad = True), std = 0.2)
        # 

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.head_film = nn.Linear(dim, num_classes) 

        # Set weights of head film to 0
        self.head_film.weight = nn.Parameter(torch.zeros_like(self.head_film.weight))
        self.head_film.bias = nn.Parameter(torch.zeros_like(self.head_film.bias))

    def rm_nan(self, x, batch):
        if not self.nan_mask: self.nan_mask = torch.isnan(x).logical_not()
        x = x[self.nan_mask]
        return x.reshape(batch,-1,self.dim)
    
    def forward(self, img):
        # img = img[None] #?? do i need this, get a key error if i don't
        B = img.shape[0] #cringe
        # x = self.to_patch_embedding(img)

        # # class token? needs changes to the pos_embedding, add the extra token
        # b, n, _ = x.shape ## !!!! batch has to be size 1 ? The [None] is needed since the Transformer has a channel dimension
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)

        # x += self.pos_embedding[:, :(n + 1)]
        # x += self.pos_embedding.to(self.device, dtype=x.dtype) # ([1, 4050, 1024]) ->  

        # remove nan values from sst
        # x = self.rm_nan(img,b)
        x, self.nan_mask, self.nan_mask_th  = self.to_patch_embedding(img)
        # pos_embed = self.pos_embedding.to(self.device, dtype=x.dtype)
        # x += pos_embed[...,self.to_patch_embedding.mask,:] #######!!!!! batch aah, and handle this in to_patch_embedding


        pos_embed = self.encoder_position_code.expand(B, -1, -1)[...,self.nan_mask_th,:]
        x = x + pos_embed #######!!!!! batch aah, and handle this in to_patch_embedding

        # x = x[torch.isnan(x).logical_not()]
        # x = x.reshape(b,-1,self.dim)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        # heads for gamma and beta (curretly only 1 gamma/beta used across blocks)
        x = self.head_film(x)
        return x
        # return self.mlp_head(x),self.mlp_head(x)
    

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
