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
from rotate_scale import rotate_3d as rotate_3d
from rotate_scale import applyTransform as applyTransform

import dm_pix
from dm_pix._src import interpolation
from dm_pix._src import augment
import functools
from functools import partial

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
# label = sitk.GetArrayFromImage(sitk.ReadImage(dictt['label']))

# label.shape #(55, 512, 512)
# rotMat= rotate_3d(10,10,10)
# applyTransform(image, rotMat)

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
    # rotation_x = jnp.array([[1,    0,   0, 0],
    #                 [0,  rcx, rsx, 0],
    #                 [0, -rsx, rcx, 0],
    #                 [0,    0,   0, 1]])

    # rcy = jnp.cos(angle_y)
    # rsy = jnp.sin(angle_y)
    # rotation_y = jnp.array([[rcy, 0, -rsy, 0],
    #                 [  0, 1,    0, 0],
    #                 [rsy, 0,  rcy, 0],
    #                 [  0, 0,    0, 1]])

    # rcz = jnp.cos(angle_z)
    # rsz = jnp.sin(angle_z)
    # rotation_z = jnp.array([[ rcz, rsz, 0, 0],
    #                 [-rsz, rcz, 0, 0],
    #                 [   0,   0, 1, 0],
    #                 [   0,   0, 0, 1]])
    # matrix = rotation_x @ rotation_y @ rotation_z

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



Nz, Ny, Nx = image.shape

@partial(jax.jit, static_argnames=['Nz','Ny','Nx'])
def apply_affine(image,trans_mat_inv,Nz, Ny, Nx):

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


    # x_valid1 = xx_prime>=0
    # x_valid2 = xx_prime<=Nx-1
    # y_valid1 = yy_prime>=0
    # y_valid2 = yy_prime<=Ny-1
    # z_valid1 = zz_prime>=0
    # z_valid2 = zz_prime<=Nz-1
    # valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2

    # z_valid_idx, y_valid_idx, x_valid_idx = jnp.where(valid_voxel > 0)
    # bbb=jnp.where(valid_voxel > 0)
    # print(f"valid_voxel {valid_voxel.shape} bbb {bbb[0].shape}")

    # image_transformed = jnp.zeros((Nz, Ny, Nx))

    # data_w_coor = RegularGridInterpolator((z, y, x), image)
    # interp_points = jnp.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
    #         yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
    #         xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
    interpolate_function = augment._get_interpolate_function(
        mode="constant",
        order=1,
        cval=0.0,
    )    
    interp_points = jnp.array([zz_prime,yy_prime, xx_prime])#.T    
    interp_result = interpolate_function(image, interp_points) #data_w_coor(interp_points)
    return interp_result
    # image_transformed=image_transformed.at[z_valid_idx, y_valid_idx, x_valid_idx].set(interp_result)
   
    # return image_transformed

# apply_affine(image,Nz, Ny, Nx)
trans_mat_inv = jnp.linalg.inv(rotate_3d(0.0,0.09,0.0)[0:3,0:3])

image_transformed=apply_affine(image,trans_mat_inv,Nz, Ny, Nx)


image_transformed = jnp.swapaxes(image_transformed, 0,2)

toSave = sitk.GetImageFromArray(image_transformed)  
toSave.SetSpacing(imagePrim.GetSpacing())
toSave.SetOrigin(imagePrim.GetOrigin())
toSave.SetDirection(imagePrim.GetDirection()) 

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName('/workspaces/Jax_cuda/old/sth.nii.gz')
writer.Execute(toSave)    










# interpolate_function = augment._get_interpolate_function(
#     mode="nearest",
#     order=1,
#     cval=0.0,
# )

# interp_points = jnp.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
#         yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
#         xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
# interp_result = interpolate_function(interp_points)
# image_transformed[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result

# interp_points.shape

# https://gist.github.com/Edenhofer/f248f0de5a1dce54a246375d53345bc5

# import dm_pix
# from dm_pix._src import interpolation
# from dm_pix._src import augment


# meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
#                         indexing="ij")
# indices = jnp.concatenate(
#     [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)


# def rotate_3d(angle_x=0, angle_y=0, angle_z=0):
#     """
#     Returns transformation matrix for 3d rotation.
#     Args:
#         angle_x: rotation angle around x axis in radians
#         angle_y: rotation angle around y axis in radians
#         angle_z: rotation angle around z axis in radians
#     Returns:
#         A 4x4 float32 transformation matrix.
#     """
#     rcx = jnp.cos(angle_x)
#     rsx = jnp.sin(angle_x)
#     rotation_x = jnp.array([[1,    0,   0, 0],
#                     [0,  rcx, rsx, 0],
#                     [0, -rsx, rcx, 0],
#                     [0,    0,   0, 1]])

#     # rcy = jnp.cos(angle_y)
#     # rsy = jnp.sin(angle_y)
#     # rotation_y = jnp.array([[rcy, 0, -rsy, 0],
#     #                 [  0, 1,    0, 0],
#     #                 [rsy, 0,  rcy, 0],
#     #                 [  0, 0,    0, 1]])

#     # rcz = jnp.cos(angle_z)
#     # rsz = jnp.sin(angle_z)
#     # rotation_z = jnp.array([[ rcz, rsz, 0, 0],
#     #                 [-rsz, rcz, 0, 0],
#     #                 [   0,   0, 1, 0],
#     #                 [   0,   0, 0, 1]])
#     # matrix = rotation_x @ rotation_y @ rotation_z
#     # return matrix
#     return rotation_x


# def shear_3d(sxy=0., sxz=0., syx=0., syz=0., szx=0., szy=0.):
#     """
#     Returns transformation matrix for 3d shearing.
#     Args:
#         sxy: xy shearing factor
#         sxz: xz shearing factor
#         syx: yx shearing factor
#         syz: yz shearing factor
#         szx: zx shearing factor
#         szy: zy shearing factor
#     Returns:
#         A 4x4 float32 transformation matrix.
#     """
#     return jnp.array([[  1, sxy, sxz],
#                         [syx,   1, syz],
#                         [szx, szy,   1]
#                         ], dtype='float32')


# matr = rotate_3d(1.0,0.0,0.0)[0:3,0:3]
# image_center = (jnp.asarray(image.shape) - 1.) / 2.
# offset = image_center - matr @ image_center
# offset = jnp.full((3,), fill_value=offset)
# matr=jnp.linalg.inv(matr)
# coordinates = indices @ matr[0:3,0:3]
# coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)
# coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

# interpolate_function = augment._get_interpolate_function(
#     mode="nearest",
#     order=1,
#     cval=0.0,
# )
# res=interpolate_function(image, coordinates)
# res.shape


# res = jnp.swapaxes(res, 0,2)

# toSave = sitk.GetImageFromArray(res)  
# toSave.SetSpacing(imagePrim.GetSpacing())
# toSave.SetOrigin(imagePrim.GetOrigin())
# toSave.SetDirection(imagePrim.GetDirection()) 

# writer = sitk.ImageFileWriter()
# writer.KeepOriginalImageUIDOn()
# writer.SetFileName('/workspaces/Jax_cuda/old/sth.nii.gz')
# writer.Execute(toSave)    

