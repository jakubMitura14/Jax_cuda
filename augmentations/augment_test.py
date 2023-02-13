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
import simpleTransforms
from simpleTransforms import elastic_deformation
from simpleTransforms import apply_rotation

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
image = jnp.swapaxes(image, 0,2)# for itk


param_x=jnp.array([[4,0.1],[6,0.01]])
param_y=jnp.array([[4,0.1],[6,0.01]])
param_z=jnp.array([[2,0.01],[2,0.001]])


image = einops.rearrange(image,'h w d -> h w d 1')
image_transformed=elastic_deformation(image,param_x,param_y,param_z)
print(f" image_transformed aa  {image_transformed.shape} ")

image_transformed=apply_rotation(image_transformed)
# image_transformed=simpleTransforms.apply_rotation_single_chan(image_transformed[:,:,:,0])

print(f" image_transformed {image_transformed.shape} ")



##transforming just to get back waht itk likes
image_transformed = jnp.swapaxes(image_transformed, 0,2)
toSave = sitk.GetImageFromArray(image_transformed)  
toSave.SetSpacing(imagePrim.GetSpacing())
toSave.SetOrigin(imagePrim.GetOrigin())
toSave.SetDirection(imagePrim.GetDirection()) 

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName('/workspaces/Jax_cuda/old/sth.nii.gz')
writer.Execute(toSave)    
