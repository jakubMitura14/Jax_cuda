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


# create the scaling transformation matrix 2D
transform_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
image=np.random.random((10,15,20))

trans_mat_inv = np.linalg.inv(transform_matrix)
Nz, Ny, Nx = image.shape
x = np.linspace(0, Nx - 1, Nx)
y = np.linspace(0, Ny - 1, Ny)
z = np.linspace(0, Nz - 1, Nz)
zza, yy, xx = np.meshgrid(z, y, x, indexing='ij')

xx,yy,zz = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
                        indexing="ij")
zza==zz

zz.shape


coor = np.array([xx - x_center, yy - y_center, zz - z_center])
coor_prime = numpy.tensordot(trans_mat_inv, coor, axes=((1), (0)))
xx_prime = coor_prime[0] + x_center
yy_prime = coor_prime[1] + y_center
zz_prime = coor_prime[2] + z_center

x_valid1 = xx_prime>=0
x_valid2 = xx_prime<=Nx-1
y_valid1 = yy_prime>=0
y_valid2 = yy_prime<=Ny-1
z_valid1 = zz_prime>=0
z_valid2 = zz_prime<=Nz-1
valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
z_valid_idx, y_valid_idx, x_valid_idx = np.where(valid_voxel > 0)

image_transformed = numpy.zeros((Nz, Ny, Nx))

data_w_coor = RegularGridInterpolator((z, y, x), image, method=method)
interp_points = numpy.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
        yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
        xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
interp_result = data_w_coor(interp_points)
image_transformed[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result
