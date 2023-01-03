from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops

import jax
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops


# x=jnp.ones((2,16,16,16,3))
# x0 = x[:, 0::2, 0::2, 0::2, :]
# x1 = x[:, 1::2, 0::2, 0::2, :]
# x2 = x[:, 0::2, 1::2, 0::2, :]
# x3 = x[:, 0::2, 0::2, 1::2, :]
# x4 = x[:, 1::2, 0::2, 1::2, :]
# x5 = x[:, 0::2, 1::2, 0::2, :]
# x6 = x[:, 0::2, 0::2, 1::2, :]
# x7 = x[:, 1::2, 1::2, 1::2, :]
# x = jnp.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1) 
# x.shape #(2, 8, 8, 8, 24)

# xb=jnp.ones((2,16,16,16,3))
# einops.rearrange(xb, 'a b c d e-> a b c d e')


import my_jax_3d
from my_jax_3d import SwinTransformer

prng = jax.random.PRNGKey(42)

feature_size  = 24
in_chans=1
depths= (2, 2, 2, 2)
num_heads = (3, 6, 12, 24)
patch_size = (2,2,2)
window_size = (8,8,8)
img_size = (1,1,32,32,32)




# monaiSwin= monai_swin_nD.SwinTransformer(in_chans=1
#                         ,embed_dim=feature_size
#                         ,window_size=window_size
#                         ,patch_size=patch_size
#                         ,depths=depths
#                         ,num_heads=num_heads                        
#                         )
# aa=monaiSwin.forward(torch.ones(img_size))
# print(f"monai shape {aa[2].shape}")#[1, 24, 8, 8, 8]

jax_swin= my_jax_3d.SwinTransformer(img_size=img_size
                    ,in_chans=in_chans
                    ,embed_dim=feature_size
                    ,window_size=window_size
                    ,patch_size=patch_size
                    ,depths=depths
                    ,num_heads=num_heads                           
                    )

input=jnp.ones(img_size)
params = jax_swin.init(prng, input,train=False)['params'] # initialize parameters by passing a template image
bb= jax_swin.apply({'params': params},input,train=False)
print(f"jax shapee 0 {bb[0].shape}  1 {bb[1].shape} 2 {bb[2].shape} 3 {bb[3].shape} ")#[1, 24, 8, 8, 8]


# monPatchEmbed = monai_swin_nD.PatchEmbed(patch_size=patch_size
#                                         ,in_chans=in_chans
#                                         ,embed_dim=feature_size )

# aa=monPatchEmbed.forward(torch.ones(img_size))
# print(f"aa {aa.shape}  ")#[1, 24, 8, 8, 8]
# prng = jax.random.PRNGKey(42)
# jaxPatchEmbed = my_jax_3d.PatchEmbed(img_size=img_size
#                                     ,patch_size=patch_size
#                                     ,embed_dim=feature_size
#                                     ,in_channels=in_chans
#                                     )

# input = jnp.ones(img_size)

# params = jaxPatchEmbed.init(prng, input)['params'] # initialize parameters by passing a template image
# bb= jaxPatchEmbed.apply({'params': params},input)
# print(f"bb {bb.shape}  ")#[1, 24, 8, 8, 8]

# tf.config.experimental.set_visible_devices([], 'GPU')


# prng = jax.random.PRNGKey(42)
# initializer = jax.nn.initializers.glorot_normal()#xavier initialization
# swin = SwinTransformer()
# params = swin.init(prng, jnp.ones([1, 8,8,8, 1]))['params'] # initialize parameters by passing a template image

# import torch
# import jax
# from jax import lax, random, numpy as jnp
# import numpy as np
# len = np.product([4, 5, 5, 5, 96])
# aaa= jnp.arange(len)
# aj=aaa.reshape((4, 5, 5, 5, 96) )
# at=torch.from_numpy(np.array(aj))

# np.array_equal(aj,at)


# atb=at.flatten(2).transpose(1, 2) #[4, 2400, 5]

# shh = aj.shape
# sss=aj.reshape(shh[0],shh[1],-1)
# ajb=jnp.swapaxes(sss,1,2)

# np.array_equal(np.array(atb),np.array(ajb))
coords_d = jnp.arange(window_size[0])
coords_h = jnp.arange(window_size[1])
coords_w = jnp.arange(window_size[2])
coords = jnp.stack(jnp.meshgrid(coords_d, coords_h, coords_w))
sss = coords.shape
tt= torch.zeros(sss)
torch.flatten(tt, 1).shape
sss

einops.rearrange(coords,'d a b c -> d (a b c)').shape