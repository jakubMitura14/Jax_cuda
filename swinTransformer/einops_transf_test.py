from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
from einops import rearrange
import jax 
from flax.linen import partitioning as nn_partitioning
import jax
import torch

feature_size  = 24 #by how long vector each image patch will be represented
in_chans=1

patch_size = (2,2,2)
window_size = (2,2,2) # in my definition it is number of patches it holds
img_size = (1,1,32,32,16)



class RelativePositionBias3D(nn.Module):
    """
    based on https://github.com/HEEHWANWANG/ABCD-3DCNN/blob/7b4dc0e132facfdd116ceb42eb026119a1a66e35/STEP_3_Self-Supervised-Learning/MAE_DDP/util/pos_embed.py

    """
    dim: int
    num_heads: int
    window_size: Tuple[int]

    def get_rel_pos_index(self):
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1) + 3

        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(window_size[0])
        coords_w = jnp.arange(window_size[1])
        coords_d = jnp.arange(window_size[2])

        coords = jnp.stack(jnp.meshgrid([coords_h, coords_w, coords_d], indexing="ijk"))  # 3, Wh, Ww, Wd
        coords_flatten = jnp.reshape(coords, (2, -1))  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = jnp.transpose(relative_coords,(1, 2, 0))  # Wd*Wh*Ww, Wd*Wh*Ww, 3
       
        relative_coords[:, :, 0].add(self.window_size[0] - 1) # shift to start from 0
        relative_coords[:, :, 1].add(self.window_size[1] - 1)
        relative_coords[:, :, 2].add(self.window_size[2] - 1)
        relative_coords[:, :, 0].multiply((2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1].multiply((2 * self.window_size[2] - 1))
        
        #so we ae here summing all of the relative distances in x y and z axes
        #TODO experiment with multiplyinstead of add but in this case do not shit above to 0 
        relative_pos_index = jnp.sum(relative_coords, -1)
        return relative_pos_index



    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=None):

        #initializing hyperparameters and parameters (becouse we are in compact ...)
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        rpbt = self.param(
            "relative_position_bias_table",
            nn.initializers.normal(0.02),
            (
                self.num_relative_distance,
                self.num_heads,
            ),
        )
        relative_pos_index = self.variable(
            "relative_position_index", "relative_position_index", self.get_rel_pos_index
        )

        rel_pos_bias = jnp.reshape(
            rpbt[jnp.reshape(relative_pos_index.value, (-1))],
            (
                self.window_size[0] * self.window_size[1] * self.window_size[2] + 1,
                self.window_size[0] * self.window_size[1] * self.window_size[2] + 1, -1
            ),
        )
        rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
        return rel_pos_bias































# class PatchMerging(nn.Module):
#     """The `PatchMerging` module previously defined in v0.9.0."""
#     dim:int
#     spatial_dims:int = 3
#     norm_layer :Type[nn.LayerNorm] = nn.LayerNorm

       
#     def setup(self):
#         self.reduction = nn.Dense(features=2*self.dim, use_bias=False)
#         self.norm = self.norm_layer()
    
#     @nn.compact
#     def __call__(self, x):
#         x_shape = x.shape
#         if len(x_shape) != 5:
#             raise ValueError(f"expecting 5D x, got {x.shape}.")
#         b, d, h, w, c = x_shape
#         x = x.reshape(b, h, w,d, c)
#         # pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
#         # if pad_input:
#         #     x = jnp.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
#         x0 = x[:, 0::2, 0::2, 0::2, :]
#         x1 = x[:, 1::2, 0::2, 0::2, :]
#         x2 = x[:, 0::2, 1::2, 0::2, :]
#         x3 = x[:, 0::2, 0::2, 1::2, :]
#         x4 = x[:, 1::2, 0::2, 1::2, :]
#         x5 = x[:, 0::2, 1::2, 0::2, :]
#         x6 = x[:, 0::2, 0::2, 1::2, :]
#         x7 = x[:, 1::2, 1::2, 1::2, :]
#         x = jnp.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)  # B H/2 W/2 4*C
#         #x = jnp.concatenate([x0, x0, x0, x0, x0, x0, x0, x0], axis=-1)  # B H/2 W/2 4*C
#         x = self.norm(x)
#         x = self.reduction(x)
#         return x

# img_size = (1,8,8,8,1)
# lenn=np.product((list(img_size )))
# x=jnp.arange(lenn)
# x= x.reshape(img_size)
# x0 = x[:, 0::2, 0::2, 0::2, :]
# x1 = x[:, 1::2, 0::2, 0::2, :]
# x2 = x[:, 0::2, 1::2, 0::2, :]
# x3 = x[:, 0::2, 0::2, 1::2, :]
# x4 = x[:, 1::2, 0::2, 1::2, :]
# x5 = x[:, 0::2, 1::2, 0::2, :]
# x6 = x[:, 0::2, 0::2, 1::2, :]
# x7 = x[:, 1::2, 1::2, 1::2, :]
# x = jnp.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)  # B H/2 W/2 4*C        
# x.shape

# x

# ims=jnp.arange(lenn)
# ims= ims.reshape(img_size)

# aa=rearrange(ims, 'b (c1 d) (c2 h) (c3 w) c -> b d h w (c1 c2 c3 c) ', c1=2, c2=2, c3=2)
# bb=rearrange(ims, 'b (c1 d) (c2 h) (c3 w) c -> b d h w (c c3 c2 c1) ', c1=2, c2=2, c3=2)
# aa
# x[0,:,0,0,0]


# ww=np.array(bb[0,0,0,:,0])
# print(ww)

# aa.shape
# jnp.array_equal( x,aa )

# x
# aa