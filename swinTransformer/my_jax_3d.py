#https://github.com/minyoungpark1/swin_transformer_v2_jax
#https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
###  krowa https://www.researchgate.net/publication/366213226_Position_Embedding_Needs_an_Independent_Layer_Normalization
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

def window_partition(x, window_size):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    b, d, h, w, c = x_shape
    x = x.reshape(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c
    )
    windows = jnp.transpose(x,  (0, 1, 3, 5, 2, 4, 6, 7)).reshape(-1, window_size[0] * window_size[1] * window_size[2], c)

    return windows

def window_reverse(windows, window_size, dims):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    b, d, h, w = dims
    x = windows.reshape(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7)).reshape(b, d, h, w, -1)
    return x
class MLP(nn.Module):
    """
    based on https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
    Transformer MLP / feed-forward block.
    both hidden and out dims are ints

    """
    hidden_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                    Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)
    act_layer: Optional[Type[nn.Module]] = nn.gelu

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)
        # x = nn.gelu(x)
        x = self.act_layer(x)
        x = self.dropout(x, deterministic=deterministic)
        x = nn.Dense(features=actual_out_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init, 
                     bias_init=self.bias_init)(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

def get_window_size(x_size, window_size, shift_size=None):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)        

class WindowAttention(nn.Module):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    based on https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py

    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    dim: int
    window_size: Tuple[int]
    num_heads: int
    qkv_bias: Optional[bool] = True
    qk_scale: Optional[float] = None
    attn_drop_rate: Optional[float] = 0.0
    proj_drop_rate: Optional[float] = 0.0

    def get_relative_coords(self,window_size):
        """
        get the relative positions of the entries inside the window
        it depends only on the shape of the window
        """
        relative_coords_h = np.arange(-(window_size[0] - 1), window_size[0], dtype=np.float32)
        relative_coords_w = np.arange(-(window_size[1] - 1), window_size[1], dtype=np.float32)
        relative_coords_d = np.arange(-(window_size[2] - 1), window_size[2], dtype=np.float32)
        
        relative_coords_table = np.expand_dims(np.transpose(np.stack(
            np.meshgrid(relative_coords_h,
                            relative_coords_w,relative_coords_d)), (1, 2, 0)), axis=0)  # 1, 2*Wh-1, 2*Ww-1, 2
        #Haven't implemented retraining a model with a different pretrained window size
        relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)
        relative_coords_table[:, :, :, 2] /= (window_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        # still using log but log2 in this case
        return np.sign(relative_coords_table) * np.log2(
            np.abs(relative_coords_table) + 1.0) / np.log2(8)  

    def setup(self):
        self.cpb = MLP(hidden_dim=512,
                       out_dim=self.num_heads,
                       dropout_rate=0.0,
                       act_layer=nn.relu)
        #used later as maximum allowed value of attention as far as I get it               
        self.logit_scale = self.param('tau', nn.initializers.normal(0.02), (1,self.num_heads, 1, 1)) + jnp.log(10)
        if self.qkv_bias:
            # keys are ignored
            self.q_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias, 
                                    bias_init=nn.initializers.constant(0))
            self.v_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias, 
                                    bias_init=nn.initializers.constant(0))
        else:
            self.q_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias)
            self.v_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias)



        # still using log but log2 in this case
        self.relative_coords_table =self.get_relative_coords(self.window_size)


        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_d = np.arange(self.window_size[1])
        x, y,z = np.meshgrid(coords_h, coords_w,coords_d)
        coords = np.stack([y, x,z])  # 3, Wh, Ww
        coords_flatten = coords.reshape(3, -1)  # 3, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        self.relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Wwbn 
          
        # self.qkv = nn.Dense(features=self.dim*3, use_bias=qkv_bias)
        self.k_linear = nn.Dense(features=self.dim, use_bias=False)
        self.attn_drop = nn.Dropout(rate=self.attn_drop_rate)
        self.proj = nn.Dense(features=self.dim)
        self.proj_drop = nn.Dropout(rate=self.proj_drop_rate)

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic):
    # def __call__(self, x, log_relative_position_index, *, mask=None, deterministic):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        #embedding and reshaping queries keys and values
        q = jnp.transpose(self.q_linear(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))[0]
        k = jnp.transpose(self.k_linear(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))[0]
        v = jnp.transpose(self.v_linear(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))[0]


        # cosine attention
        qk = jnp.clip(jnp.expand_dims(jnp.linalg.norm(q, axis=-1), axis=-1)@jnp.expand_dims(jnp.linalg.norm(k, axis=-1), axis=-2), a_min=1e-6)
        attn = q@(jnp.swapaxes(k, -2,-1))/qk
        attn = attn*jnp.clip(self.logit_scale, a_min=1e-2)

        # Log-CPB
        #nH seem to be dimension for diffrent windows
        relative_position_bias_table = self.cpb(self.relative_coords_table, deterministic=True).reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.flatten()].reshape(
            self.window_size[0] * self.window_size[1]* self.window_size[2], self.window_size[0] * self.window_size[1]** self.window_size[2], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww*Wd, Wh*Ww*Wd
        relative_position_bias = 16 * nn.sigmoid(relative_position_bias)
        attn = attn + jnp.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + jnp.expand_dims(mask, axis=(0,2))
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = nn.softmax(attn, axis=-1)
        else:
            attn = nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn, deterministic=deterministic)

        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)

        return x

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        return x

class AdaptiveAvgPool1d(nn.Module):
    """ 
        Applying a 1D adaptive average pooling over an input data.
    """
    output_size: int = 1

    @nn.compact
    def __call__(self, x):
        stride = (x.shape[1]//self.output_size)
        kernel_size = (x.shape[1]-(self.output_size-1)*stride)
        avg_pool = nn.avg_pool(inputs=x, window_shape=(kernel_size,), strides=(stride,))
        return avg_pool

def create_attn_mask(dims,shift_size, window_size):
    """
    as far as I see attention masks are needed to deal with the changing windows
    """
    d, h, w = dims
    if shift_size > 0:
        img_mask = jnp.zeros((1, d, h, w, 1))
        #we are taking into account the size of window and a shift - so we need to mask all that get out of original image
        cnt = 0
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        #so we get the matrix where dim 0 is of len= number of windows and dim 1 the flattened window
        mask_windows = mask_windows.reshape(-1, window_size[0] * window_size[1])
        attn_mask = jnp.expand_dims(mask_windows, axis=1) - jnp.expand_dims(mask_windows, axis=2)
        attn_mask = jnp.where(attn_mask==0, x=float(0.0), y=float(-100.0))
    else:
        attn_mask = None

    return attn_mask


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size Tuple[int]: Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float, optional): Dropout rate. Default: 0.0
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """


    dim: int
    input_resolution: Tuple[int]
    num_heads: int
    window_size: Tuple[int]
    shift_size: int = 0
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0
    act_layer: Type[nn.Module] = nn.gelu
    norm_layer: Type[nn.Module] = nn.LayerNorm

    def setup(self):
        self.norm1 = self.norm_layer(self.dim)
        self.attn = WindowAttention(
            self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.drop_path_rate,
        )

        self.batch_dropout = nn.Dropout(rate=self.drop_path_rate, broadcast_dims=[1,2]) \
        if self.drop_path_rate > 0. else IdentityLayer()
        self.norm2 = self.norm_layer()
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, dropout_rate=self.drop_rate,
                       act_layer=self.act_layer)

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        # pad_l = pad_t = pad_d0 = 0
        # pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        # pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        # pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = jnp.roll(x, shift=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = jnp.roll(shifted_x, shift=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    @nn.compact
    def __call__(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    """The `PatchMerging` module previously defined in v0.9.0."""
    input_resolution: Tuple[int]
    dim: int
    norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm
    
    def setup(self):
        self.reduction = nn.Dense(features=2*self.dim, use_bias=False)
        self.norm = self.norm_layer()
    
    @nn.compact
    def __call__(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        # pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x



class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    dim: int
    input_resolution: Tuple[int]
    depth: int
    num_heads: int
    window_size: Tuple[int]
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0
    norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm
    downsample: Type[nn.Module] = None# can be patch merging

    def setup(self):
        self.blocks = [SwinTransformerBlock(dim=self.dim, input_resolution=self.input_resolution,
                                            num_heads=self.num_heads, window_size=self.window_size,
                                            shift_size=0 if (i % 2 == 0) else min(self.window_size) // 2,
                                            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                                            drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate,
                                            drop_path_rate=self.drop_path_rate[i] \
                                            if isinstance(self.drop_path_rate, tuple) else self.drop_path_rate,
                                            norm_layer=self.norm_layer) 
        for i in range(self.depth)]

        # patch merging layer
        if self.downsample is not None:
            self.downsample_module = self.downsample(self.input_resolution, dim=self.dim, norm_layer=self.norm_layer)
        else:
            self.downsample_module = None
    def forward(self, x):
        x_shape = x.size()
        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = einops.rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = create_attn_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = einops.rearrange(x, "b d h w c -> b c d h w")
        return x



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (Tuple[int]): Image size.  Default: (224, 224).
        patch_size (Tuple[int]): Patch token size. Default: (4, 4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    embed_dim: int = 96
    norm_layer: Optional[Type[nn.Module]] = None
    proj = nn.Conv(features=embed_dim, 
                                kernel_size=patch_size, 
                                strides=patch_size)      
    norm = norm_layer(embed_dim)


    def forward(self, x):
        x_shape = x.size()
        _, _, d, h, w = x_shape
        if w % self.patch_size[2] != 0:
            x = jnp.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = jnp.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = jnp.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))


        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
            x = x.transpose(1, 2).reshape(-1, self.embed_dim, d, wh, ww)

        return x

class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    patch_size: Tuple[int] = (4, 4)
    in_chans: int = 3
    embed_dim: int = 96
    depths: Tuple[int] = (2, 2, 6, 2)
    num_heads: Tuple[int] = (3, 6, 12, 24)
    window_size: Tuple[int] = (7, 7)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    patch_norm: bool = False
    use_checkpoint: bool = False
    spatial_dims: int = 3
    downsample="merging"
    norm_layer: Type[nn.Module] = nn.LayerNorm

    num_layers = len(depths)
    embed_dim = embed_dim
    patch_norm = patch_norm
    window_size = window_size
    patch_size = patch_size
    patch_embed = PatchEmbed(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer if patch_norm else None,  # type: ignore
        spatial_dims=spatial_dims,
    )
    pos_drop = nn.Dropout(p=drop_rate)
    dpr = [x.item() for x in jnp.linspace(0, drop_path_rate, sum(depths))]
    layers1 = nn.ModuleList()
    layers2 = nn.ModuleList()
    layers3 = nn.ModuleList()
    layers4 = nn.ModuleList()
    down_sample_mod = PatchMerging
    for i_layer in range(num_layers):
        layer = BasicLayer(
            dim=int(embed_dim * 2**i_layer),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            downsample=down_sample_mod,
            use_checkpoint=use_checkpoint,
        )
        if i_layer == 0:
            layers1.append(layer)
        elif i_layer == 1:
            layers2.append(layer)
        elif i_layer == 2:
            layers3.append(layer)
        elif i_layer == 3:
            layers4.append(layer)
    num_features = int(embed_dim * 2 ** (num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = einops.rearrange(x, "n c d h w -> n d h w c")
                x = nn.layer_norm(x, [ch])
                x = einops.rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = einops.rearrange(x, "n c h w -> n h w c")
                x = nn.layer_norm(x, [ch])
                x = einops.rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]