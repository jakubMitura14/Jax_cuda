import jax.numpy as jnp
import flax.linen as nn



from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size Tuple[int]: window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    divides the image into windows     
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = jnp.transpose(x,  (0, 1, 3, 2, 4, 5)).reshape(-1, window_size[0], window_size[1], C)
    return windows


x= jnp.ones((1,4,4,3))

window_size=(2,2)
mask_windows = window_partition(x,window_size )
mask_windows = mask_windows.reshape(-1, window_size[0] * window_size[1])
mask_windows.shape

coords_h = jnp.arange(window_size[0])
coords_w = jnp.arange(window_size[1])
jnp.stack(jnp.meshgrid(coords_h, coords_w, coords_w))







window_size=(3,2)

relative_coords_h = np.arange(-(window_size[0] - 1), window_size[0], dtype=np.float32)#[-2., -1.,  0.,  1.,  2.]
relative_coords_w = np.arange(-(window_size[1] - 1), window_size[1], dtype=np.float32)#[-1.,  0.,  1.]

a,b=np.meshgrid(relative_coords_h,relative_coords_w)

relative_coords_table = np.expand_dims(np.transpose(np.stack(
    np.meshgrid(relative_coords_h,
                    relative_coords_w)), (1, 2, 0)), axis=0)  # 1, 2*Wh-1, 2*Ww-1, 2

# Haven't implemented retraining a model with a different pretrained window size
relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)

relative_coords_table *= 8  # normalize to -8, 8
# still using log but log2 in this case
relative_coords_table = np.sign(relative_coords_table) * np.log2(
    np.abs(relative_coords_table) + 1.0) / np.log2(8)

relative_coords_table.shape#(1,5,3,2)

coords_h = np.arange(window_size[0])
coords_w = np.arange(window_size[1])
x, y = np.meshgrid(coords_h, coords_w)
coords = np.stack([y, x])  # 2, Wh, Ww
coords_flatten = coords.reshape(2, -1)  # 2, Wh*Ww
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += window_size[1] - 1
relative_coords[:, :, 0] *= 2 * window_size[1] - 1
relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Wwbn 
relative_position_index#(6,6)


# def window_partition(x, window_size):
#     batch, height, width, channels = x.shape
#     x = jnp.reshape(
#         x,
#         (
#             batch,
#             height // window_size,
#             window_size,
#             width // window_size,
#             window_size,
#             channels,
#         ),
#     )
#     windows = jnp.reshape(
#         jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, channels)
#     )
#     return windows

# def window_reverse(windows, window_size, height, width):
#     batch = int(windows.shape[0] / (height * width / window_size / window_size))
#     x = jnp.reshape(
#         windows,
#         (
#             batch,
#             height // window_size,
#             width // window_size,
#             window_size,
#             window_size,
#             -1,
#         ),
#     )
#     x = jnp.reshape(jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (batch, height, width, -1))
#     return x

# ############# WindowAttention

# def get_rel_pos_index(window_size_tupl):
#     """
#     window is a tuple we return sth like a distance from top right corner to the 
#     bottom left corner - I suppose it is related with relative positional embedding
#     """
#     h_indices = jnp.arange(0, window_size_tupl[0])
#     w_indices = jnp.arange(0, window_size_tupl[1])
#     indices = jnp.stack(jnp.meshgrid(w_indices, h_indices, indexing="ij"))
#     flatten_indices = jnp.reshape(indices, (2, -1))
#     relative_indices = flatten_indices[:, :, None] - flatten_indices[:, None, :]
#     relative_indices = jnp.transpose(relative_indices, (1, 2, 0))
#     relative_indices = relative_indices.at[:, :, 0].add(window_size_tupl[0] - 1)
#     relative_indices = relative_indices.at[:, :, 1].add(window_size_tupl[1] - 1)
#     relative_indices = relative_indices.at[:, :, 0].multiply(
#         2 * window_size_tupl[1] - 1
#     )
#     relative_pos_index = jnp.sum(relative_indices, -1)
#     return relative_pos_index



# # mergeparam used when hyperparameter can be passed both in init and call
# deterministic = nn.merge_param(
#     "deterministic", self.deterministic, deterministic
# )

# rpbt = self.param(
#     "relative_position_bias_table",
#     nn.initializers.normal(0.02),
#     (
#         (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
#         self.num_heads,
#     ),
# )

# relative_pos_index = self.variable(
#     "relative_position_index", "relative_position_index", self.get_rel_pos_index
# )

# batch, n, channels = inputs.shape
# qkv = nn.Dense(self.dim * 3, use_bias=self.use_bias, name="qkv")(inputs)
# qkv = qkv.reshape(batch, n, 3, self.num_heads, channels // self.num_heads)
# qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
# q, k, v = qkv[0], qkv[1], qkv[2]

# q = q * ((self.dim // self.num_heads) ** -0.5)
# att = q @ jnp.swapaxes(k, -2, -1)

# rel_pos_bias = jnp.reshape(
#     rpbt[jnp.reshape(relative_pos_index.value, (-1))],
#     (
#         self.window_size[0] * self.window_size[1],
#         self.window_size[0] * self.window_size[1],
#         -1,
#     ),
# )
# rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
# att += jnp.expand_dims(rel_pos_bias, 0)

# if mask is not None:
#     att = jnp.reshape(
#         att, (batch // mask.shape[0], mask.shape[0], self.num_heads, n, n)
#     )
#     att = att + jnp.expand_dims(jnp.expand_dims(mask, 1), 0)
#     att = jnp.reshape(att, (-1, self.num_heads, n, n))
#     att = nn.softmax(att)

# else:
#     att = nn.softmax(att)

# att = nn.Dropout(self.att_drop)(att, deterministic)

# x = jnp.reshape(jnp.swapaxes(att @ v, 1, 2), (batch, n, channels))
# x = nn.Dense(self.dim, name="proj")(x)
# x = nn.Dropout(self.proj_drop)(x, deterministic)
# return x


# get_rel_pos_index((2,2))


# import numpy as np
# ar=np.random.rand(16)
# ar = jnp.arange(4)
# ar= jnp.reshape(ar,(2,2))
# jnp.expand_dims(ar, 1)[1][1][2]
# jnp.expand_dims(ar, 2)[1][1][2]
# ee=jnp.expand_dims(ar, 1) - jnp.expand_dims(ar, 2)
# ee

# img =jnp.ones((1,8,16,3))
# aa=window_partition(img, 4)
# bb= window_reverse()
# aa.shape

