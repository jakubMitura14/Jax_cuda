from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import my_jax_3d
from my_jax_3d import SwinTransformer
import jax
import tensorflow as tf
import monai_swin_nD
import torch 

prng = jax.random.PRNGKey(42)

feature_size  = 24
in_chans=1
depths= (2, 2, 2, 2)
num_heads = (3, 6, 12, 24)
patch_size = (4,4,4)
window_size = (7,7,7)
img_size = (1,1,16,16,16)

# monaiSwin= monai_swin_nD.SwinTransformer(in_chans=1
#                         ,embed_dim=feature_size
#                         ,window_size=window_size
#                         ,patch_size=patch_size
#                         ,depths=depths
#                         ,num_heads=num_heads                        
#                         )
# aa=monaiSwin.forward(torch.ones(img_size))
# print(f"monai shape {aa.shape}")#[1, 24, 8, 8, 8]

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
print(f"jax shapee {bb.shape} ")#[1, 24, 8, 8, 8]


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
