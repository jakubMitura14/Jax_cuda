from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
# import monai_swin_nD
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
# import rotate_scale as rotate_scale
import SimpleITK as sitk

import dm_pix
from dm_pix._src import interpolation
from dm_pix._src import augment
import functools
from functools import partial


@jax.jit
def rotate_3d(angle_x=0.0, angle_y=0.0, angle_z=0.0):
    """
    Returns transformation matrix for 3d rotation.
    Args:
        angle_x: rotation angle around x axis in radians
        angle_y: rotation angle around y axis in radians
        angle_z: rotation angle around z axis in radians
    Returns:
        A 4x4 float32 transformation matrix.
    """
    rcx = jnp.cos(angle_x)
    rsx = jnp.sin(angle_x)
    rotation_x = jnp.array([[1,    0,   0, 0],
                    [0,  rcx, -rsx, 0],
                    [0, rsx, rcx, 0],
                    [0,    0,   0, 1]])

    rcy = jnp.cos(angle_y)
    rsy = jnp.sin(angle_y)
    rotation_y = jnp.array([[rcy, 0, rsy, 0],
                    [  0, 1,    0, 0],
                    [-rsy, 0,  rcy, 0],
                    [  0, 0,    0, 1]])

    rcz = jnp.cos(angle_z)
    rsz = jnp.sin(angle_z)
    rotation_z = jnp.array([[ rcz, -rsz, 0, 0],
                    [rsz, rcz, 0, 0],
                    [   0,   0, 1, 0],
                    [   0,   0, 0, 1]])
    matrix = rotation_x @ rotation_y @ rotation_z

    return matrix


@partial(jax.jit, static_argnames=['Nz','Ny','Nx'])
def apply_affine_rotation(image,trans_mat_inv,Nz, Ny, Nx):

    x = jnp.linspace(0, Nx - 1, Nx)
    y = jnp.linspace(0, Ny - 1, Ny)
    z = jnp.linspace(0, Nz - 1, Nz)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing='ij')
    z_center, y_center,x_center= (jnp.asarray(image.shape) - 1.) / 2.
    coor = jnp.array([xx - x_center, yy - y_center, zz - z_center])
    coor_prime = jnp.tensordot(trans_mat_inv, coor, axes=((1), (0)))
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center


    interpolate_function = augment._get_interpolate_function(
        mode="constant",
        order=1,
        cval=0.0,
    )    
    interp_points = jnp.array([zz_prime,yy_prime, xx_prime])#.T    
    interp_result = interpolate_function(image, interp_points) #data_w_coor(interp_points)
    return interp_result
    # image_transformed=image_transformed.at[z_valid_idx, y_valid_idx, x_valid_idx].set(interp_result)
   
