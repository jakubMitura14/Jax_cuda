# taken from https://github.com/4rtemi5/imax/blob/master/imax/transforms.py
import jax
import jax.numpy as jnp
I = jnp.identity(4)



@jax.jit
def rotate_3d(angle_x=0, angle_y=0, angle_z=0):
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
                    [0,  rcx, rsx, 0],
                    [0, -rsx, rcx, 0],
                    [0,    0,   0, 1]])

    rcy = jnp.cos(angle_y)
    rsy = jnp.sin(angle_y)
    rotation_y = jnp.array([[rcy, 0, -rsy, 0],
                    [  0, 1,    0, 0],
                    [rsy, 0,  rcy, 0],
                    [  0, 0,    0, 1]])

    rcz = jnp.cos(angle_z)
    rsz = jnp.sin(angle_z)
    rotation_z = jnp.array([[ rcz, rsz, 0, 0],
                    [-rsz, rcz, 0, 0],
                    [   0,   0, 1, 0],
                    [   0,   0, 0, 1]])
    matrix = rotation_x @ rotation_y @ rotation_z
    return matrix


@jax.jit
def shear_3d(sxy=0., sxz=0., syx=0., syz=0., szx=0., szy=0.):
    """
    Returns transformation matrix for 3d shearing.
    Args:
        sxy: xy shearing factor
        sxz: xz shearing factor
        syx: yx shearing factor
        syz: yz shearing factor
        szx: zx shearing factor
        szy: zy shearing factor
    Returns:
        A 4x4 float32 transformation matrix.
    """
    matrix = jnp.array([[  1, sxy, sxz, 0],
                        [syx,   1, syz, 0],
                        [szx, szy,   1, 0],
                        [  0,   0,   0, 1]], dtype='float32')
    return matrix

@jax.jit
def scale_3d(scale_x=1., scale_y=1., scale_z=1., scale_xyz=1.):
    """
    Returns transformation matrix for 3d scaling.
    Args:
        scale_x: scaling factor in x-direction
        scale_y: scaling factor in y-direction
        scale_z: scaling factor in z-direction
        scale_xyz: scaling factor in all directions
    Returns:
        A 4x4 float32 transformation matrix.
    """
    matrix = jnp.array([[1 / scale_x, 0, 0, 0],
                        [0, 1 / scale_y, 0, 0],
                        [0, 0, 1 / scale_z, 0],
                        [0, 0, 0, 1 / scale_xyz]], dtype='float32')
    return matrix


def applyTransform(image, transform_matrix):
    return jnp.matmul(image, transform_matrix)

