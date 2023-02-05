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
import rotate_scale as rotate_scale
import SimpleITK as sitk


import dm_pix
from dm_pix._src import interpolation
from dm_pix._src import augment

import functools
from functools import partial


import functools
from typing import Callable, Sequence, Tuple, Union

import chex
from dm_pix._src import color_conversion
from dm_pix._src import interpolation
import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(42)

data_dir='/root/data'
train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

dictt=data_dicts[0]
imagePrim=sitk.ReadImage(dictt['image'])
image = sitk.GetArrayFromImage(imagePrim)
image = jnp.swapaxes(image, 0,2)

# Nx,Ny,Nz= image.shape
# fullArr=jnp.arange(Nx)
# fullArr=jnp.sin(fullArr*0.01)*200
# repX=einops.repeat(fullArr,'x->x y z 1', y=Ny, z=Nz)
# repX.shape
#### in elastic deformation the deformation size should be inversly proportional to the voxel size in this axis
# so ussually it should be smallest in z dim

def get_simple_fourier_perDim(Na,f_mul_a, f_mul_b, a_mul_a, a_mul_b):
  """
  in oder to get smoother deformations we will use here something like fourier just with only 2 waves
  a and b frequency will be controlled by f_mul_a and f_mul_b and amplitude by a_mul_a and a_mul_b 
  Na is size of the dimension that we want to currently modify
  """
  fullArr=jnp.arange(Na)
  return (jnp.sin(fullArr*f_mul_a)*a_mul_a)+(jnp.sin(fullArr*f_mul_b)*a_mul_b)
  # fullArr_a=jnp.sin(fullArr*0.1)*5



def elastic_deformation(
    key: chex.PRNGKey,
    image: chex.Array,
    alpha: chex.Numeric,
    sigma: chex.Numeric,
    *,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.,
    channel_axis: int = -1,
) -> chex.Array:
  """Applies an elastic deformation to the given image.
  Introduced by [Simard, 2003] and popularized by [Ronneberger, 2015]. Deforms
  images by moving pixels locally around using displacement fields.
  Small sigma values (< 1.) give pixelated images while higher values result
  in water like results. Alpha should be in the between x5 and x10 the value
  given for sigma for sensible resutls.
  Args:
    key: key: a JAX RNG key.
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    alpha: strength of the distortion field. Higher values mean that pixels are
      moved further with respect to the distortion field's direction.
    sigma: standard deviation of the gaussian kernel used to smooth the
      distortion fields.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0, 1]. Note that PIX interpolation will only be used
      for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
      PIX interpolation, which is very fast on accelerators (especially on
      TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
      on `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.
    channel_axis: the index of the channel axis.
  Returns:
    The transformed image.
  """

  single_channel_shape = (*image.shape[:-1], 1)
  print(f"single_channel_shape {single_channel_shape}")

  Nx,Ny,Nz,_= single_channel_shape
  arr_x=get_simple_fourier_perDim(Nx,0.1, 0.01, 4, 6)
  shift_map_i=einops.repeat(arr_x,'x->x y z 1', y=Ny, z=Nz)

  arr_y=get_simple_fourier_perDim(Ny,0.1, 0.01, 4, 6)
  shift_map_j=einops.repeat(arr_y,'y->x y z 1', x=Nx, z=Nz)

  arr_z=get_simple_fourier_perDim(Nz,0.01, 0.001, 2, 2)
  shift_map_k=einops.repeat(arr_z,'z->x y z 1', y=Ny, x=Nx)


  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in single_channel_shape],
                          indexing="ij")
  meshgrid[0] += shift_map_i
  meshgrid[1] += shift_map_j
  meshgrid[2] += shift_map_k

  interpolate_function = augment._get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  transformed_image = jnp.concatenate([
      interpolate_function(
          image[..., channel, jnp.newaxis], jnp.asarray(meshgrid))
      for channel in range(image.shape[-1])
  ], axis=-1)

  if channel_axis != -1:  # Set channel axis back to original index.
    transformed_image = jnp.moveaxis(
        transformed_image, source=-1, destination=channel_axis)
  return transformed_image



alpha = 50.0
sigma = 20.0
image = einops.rearrange(image,'h w d -> h w d 1')
image_transformed=elastic_deformation(key,image,alpha,sigma)
image_transformed = jnp.swapaxes(image_transformed, 0,2)
toSave = sitk.GetImageFromArray(image_transformed)  
toSave.SetSpacing(imagePrim.GetSpacing())
toSave.SetOrigin(imagePrim.GetOrigin())
toSave.SetDirection(imagePrim.GetDirection()) 

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName('/workspaces/Jax_cuda/old/sth.nii.gz')
writer.Execute(toSave)    
