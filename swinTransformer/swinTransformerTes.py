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

data_dir='/root/data'
train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

rang=list(range(0,len(train_images)))

subjects_list=list(map(lambda index:tio.Subject(image=tio.ScalarImage(train_images[index],),label=tio.LabelMap(train_labels[index])),rang ))
subjects_list_train=subjects_list[:-9]
subjects_list_val=subjects_list[-9:]

transforms = [
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.Resample((2.5,2.5,2.5)),
    tio.transforms.CropOrPad((256,256,128)),
    tio.RandomAffine(),
]
transform = tio.Compose(transforms)
subjects_dataset = tio.SubjectsDataset(subjects_list_train, transform=transform)


import my_jax_3d as my_jax_3d
from my_jax_3d import SwinTransformer

prng = jax.random.PRNGKey(42)

feature_size  = 24 #by how long vector each image patch will be represented
in_chans=1
depths= (2, 2, 2, 2)
num_heads = (3, 3, 3, 3)
patch_size = (4,4,4)
window_size = (4,4,4) # in my definition it is number of patches it holds
img_size = (1,1,256,256,128)

def focal_loss(inputs, targets):
    """
    based on https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """
    alpha = 0.8
    gamma = 2        
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = jax.nn.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    # inputs = inputs.view(-1)
    # targets = targets.view(-1)
    inputs=jnp.ravel(inputs)
    targets=jnp.ravel(targets)
    #first compute binary cross-entropy 
    BCE = optax.softmax_cross_entropy(inputs, targets)
    BCE_EXP = jnp.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                    
    return focal_loss

def dice_metr(y_pred,y_true):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    empty_score=1.0
    inputs = jax.nn.sigmoid(y_pred)
    inputs =inputs >= 0.5   
    im1 = inputs.astype(np.bool)
    im2 = y_true.astype(np.bool)
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = jnp.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


jax_swin= my_jax_3d.SwinTransformer(img_size=img_size
                ,in_chans=in_chans
                ,embed_dim=feature_size
                ,window_size=window_size
                ,patch_size=patch_size
                ,depths=depths
                ,num_heads=num_heads                           
                )

total_steps=700

def create_train_state(learning_rate):
    """Creates initial `TrainState`."""
    input=jnp.ones(img_size)
    params = jax_swin.init(prng, input)['params'] # initialize parameters by passing a template image
    # bb= jax_swin.apply({'params': params},input,train=False)
    # print(f"jax shapee 0 {bb.shape} ")
    warmup_exponential_decay_scheduler = optax.warmup_exponential_decay_schedule(init_value=0.001, peak_value=0.0003,
                                                                                warmup_steps=int(total_steps*0.2),
                                                                                transition_steps=total_steps,
                                                                                decay_rate=0.8,
                                                                                transition_begin=int(total_steps*0.2),
                                                                                end_value=0.0001)    
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.5),  # Clip gradients at norm 1.5
        optax.adamw(learning_rate=warmup_exponential_decay_scheduler)
    )


    return train_state.TrainState.create(
        apply_fn=jax_swin.apply, params=params, tx=tx)

state = create_train_state(0.0001)

# @nn.jit
def train_step(state, image,label,train):
  """Train for a single step."""
#   image=jnp.reshape(image,img_size )
#   res= jax_swin.apply({'params': state.params},image,train=False)
#   loss_value, grads = jax.value_and_grad(focal_loss, has_aux=False)(res,label)
#   state = state.apply_gradients(grads=grads)
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, image)
    loss = focal_loss(logits, label)
    # print(f"loss {loss} ")
    return loss, logits

  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  f_l=focal_loss(logits,label)

  return state,f_l,logits

train=True
cached_subj=[]
training_loader = DataLoader(subjects_dataset, batch_size=1, num_workers=8)
for subject in training_loader :
    cached_subj.append(subject)


for epoch in range(1, total_steps):
    dicee=0
    f_ll=0
    for subject in cached_subj :
        image=subject['image'][tio.DATA].numpy()
        label=subject['label'][tio.DATA].numpy()
        # print(f"#### {jnp.sum(label)} ")
        state,f_l,logits=train_step(state, image,label,train)
        dice=dice_metr(logits,label)
        dicee=dicee+dice
        f_ll=f_ll+f_l
    print(f"epoch {epoch} dice {dicee/len(subjects_dataset)} f_l {f_ll/len(subjects_dataset)} ")
    # print(image.shape)


# torch.backends.cudnn.version()
# 8500



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


# import my_jax_3d
# from my_jax_3d import SwinTransformer

# prng = jax.random.PRNGKey(42)

# feature_size  = 24
# in_chans=1
# depths= (2, 2, 2, 2)
# num_heads = (3, 6, 12, 24)
# patch_size = (2,2,2)
# window_size = (8,8,8)
# img_size = (1,1,64,64,32)


# jax_swin= my_jax_3d.SwinTransformer(img_size=img_size
#                     ,in_chans=in_chans
#                     ,embed_dim=feature_size
#                     ,window_size=window_size
#                     ,patch_size=patch_size
#                     ,depths=depths
#                     ,num_heads=num_heads                           
#                     )

# input=jnp.ones(img_size)
# params = jax_swin.init(prng, input,train=False)['params'] # initialize parameters by passing a template image
# bb= jax_swin.apply({'params': params},input,train=False)
# print(f"jax shapee 0 {bb.shape} ")

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
# coords_d = jnp.arange(window_size[0])
# coords_h = jnp.arange(window_size[1])
# coords_w = jnp.arange(window_size[2])
# coords = jnp.stack(jnp.meshgrid(coords_d, coords_h, coords_w))
# sss = coords.shape
# tt= torch.zeros(sss)
# torch.flatten(tt, 1).shape
# sss

# einops.rearrange(coords,'d a b c -> d (a b c)').shape