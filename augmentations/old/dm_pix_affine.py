# taken from https://github.com/deepmind/dm_pix/blob/a08d21d2d69e0ebffd49da92d16de6a6c20d45c7/dm_pix/_src/augment.py

import functools
from typing import Callable, Sequence, Tuple, Union

import chex

import jax
import jax.numpy as jnp

def affine_transform(
    image: chex.Array,
    matrix: chex.Array,
    *,
    offset: Union[chex.Array, chex.Numeric] = 0.,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
  """Applies an affine transformation given by matrix.
  Given an output image pixel index vector o, the pixel value is determined from
  the input image at position jnp.dot(matrix, o) + offset.
  This does 'pull' (or 'backward') resampling, transforming the output space to
  the input to locate data. Affine transformations are often described in the
  'push' (or 'forward') direction, transforming input to output. If you have a
  matrix for the 'push' transformation, use its inverse (jax.numpy.linalg.inv)
  in this function.
  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    matrix: the inverse coordinate transformation matrix, mapping output
      coordinates to input coordinates. If ndim is the number of dimensions of
      input, the given matrix must have one of the following shapes:
      - (ndim, ndim): the linear transformation matrix for each output
        coordinate.
      - (ndim,): assume that the 2-D transformation matrix is diagonal, with the
        diagonal specified by the given value.
      - (ndim + 1, ndim + 1): assume that the transformation is specified using
        homogeneous coordinates [1]. In this case, any value passed to offset is
        ignored.
      - (ndim, ndim + 1): as above, but the bottom row of a homogeneous
        transformation matrix is always [0, 0, 0, 1], and may be omitted.
    offset: the offset into the array where the transform is applied. If a
      float, offset is the same for each axis. If an array, offset should
      contain one value for each axis.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0-1]. Note that PIX interpolation will only be used
      for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
      PIX interpolation, which is very fast on accelerators (especially on
      TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
      on `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.
  Returns:
    The input image transformed by the given matrix.
  Example transformations:
    - Rotation:
    >>> angle = jnp.pi / 4
    >>> matrix = jnp.array([
    ...    [jnp.cos(rotation), -jnp.sin(rotation), 0],
    ...    [jnp.sin(rotation), jnp.cos(rotation), 0],
    ...    [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    - Translation: Translation can be expressed through either the matrix itself
      or the offset parameter.
    >>> matrix = jnp.array([
    ...   [1, 0, 0, 25],
    ...   [0, 1, 0, 25],
    ...   [0, 0, 1, 0],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    >>> # Or with offset:
    >>> matrix = jnp.array([
    ...   [1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> offset = jnp.array([25, 25, 0])
    >>> result = dm_pix.affine_transform(
            image=image, matrix=matrix, offset=offset)
    - Reflection:
    >>> matrix = jnp.array([
    ...   [-1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    - Scale:
    >>> matrix = jnp.array([
    ...   [2, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    - Shear:
    >>> matrix = jnp.array([
    ...   [1, 0.5, 0],
    ...   [0.5, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
  One can also combine different transformations matrices:
  >>> matrix = rotation_matrix.dot(translation_matrix)
  """
  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
                          indexing="ij")
  indices = jnp.concatenate(
      [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

  # if matrix.shape == (4, 4) or matrix.shape == (3, 4):
  #   offset = matrix[:image.ndim, image.ndim]
  #   matrix = matrix[:image.ndim, :image.ndim]

  # coordinates = indices @ matrix.T
  coordinates = indices @ matrix.T
  coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)

  # Alter coordinates to account for offset.
  offset = jnp.full((3,), fill_value=offset)
  coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

  interpolate_function = _get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  return interpolate_function(image, coordinates)