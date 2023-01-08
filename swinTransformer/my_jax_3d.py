#https://github.com/minyoungpark1/swin_transformer_v2_jax
#https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
###  krowa https://www.researchgate.net/publication/366213226_Position_Embedding_Needs_an_Independent_LayerNormalization
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax 
from flax.linen import partitioning as nn_partitioning
import jax
from einops import rearrange
from einops import einsum

remat = nn_partitioning.remat


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class DeConv3x3(nn.Module):
    """
    copied from https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/unet.py
    Deconvolution layer for upscaling.
    Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    padding: str = 'SAME'
    use_norm: bool = True
    def setup(self):  
        self.convv = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,
                
                )


    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies deconvolution with 3x3 kernel."""
        # x=einops.rearrange(x, "n c d h w -> n d h w c")
        x = self.convv(x)
        return nn.LayerNorm()(x)
           
def window_partition(input, window_size):
    """
    divides the input into partitioned windows 
    Args:
        input: input tensor.
        window_size: local window size.
    """
    return rearrange(input,'b (d w0) (h w1) (w w2) c -> (b d h w) (w0 w1 w2) c' ,w0=window_size[0],w1=window_size[1],w2= window_size[2] )#,we= window_size[0] * window_size[1] * window_size[2]  


def window_reverse(input, window_size,dims):
    """
    get from input partitioned into windows into original shape
     Args:
        input: input tensor.
        window_size: local window size.
    """
    return rearrange(input,'(b d h w) (w0 w1 w2) c -> b (d w0) (h w1) (w w2) c' 
        ,w0=window_size[0],w1=window_size[1],w2= window_size[2],b=dims[0],d=dims[1]// window_size[0],h=dims[2]// window_size[1],w=dims[3]// window_size[2] ,c=dims[4])#,we= window_size[0] * window_size[1] * window_size[2]  

class MLP(nn.Module):
    """
    based on https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
    Transformer MLP / feed-forward block.
    both hidden and out dims are ints

    """
    hidden_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                    Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)
    act_layer: Optional[Type[nn.Module]] = nn.gelu

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init,
                    #  param_dtype=jax.numpy.float16
                     )(x)
        # x = nn.gelu(x)
        x = self.act_layer(x)
        x = self.dropout(x, deterministic=deterministic)
        x = nn.Dense(features=actual_out_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init, 
                     bias_init=self.bias_init,
                    #  param_dtype=jax.numpy.float16
                     )(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

class myConv3D(nn.Module):
    """ 
        Applying a 3D convolution with variable number of channels, what is complicated in 
        Flax conv
    """
    kernel_size :Tuple[int] =(3,3,3)
    stride :Tuple[int] = (1,1,1)
    in_channels: int = 1
    out_channels: int = 1
    

    def setup(self):
        self.initializer = jax.nn.initializers.glorot_normal()#xavier initialization
        self.weight_size = (self.out_channels, self.in_channels) + self.kernel_size

    @nn.compact
    def __call__(self, x):
        parr = self.param('parr', lambda rng, shape: self.initializer(rng,self.weight_size), self.weight_size)
        return  jax.lax.conv_general_dilated(x, parr,self.stride, 'SAME')



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (Tuple[int]): Image size.  Default: (224, 224).
        patch_size (Tuple[int]): Patch token size. Default: (4, 4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    img_size: Tuple[int]
    patch_size: Tuple[int]
    embed_dim: int
    in_channels: int
    norm_layer=nn.LayerNorm
  
    def setup(self):
        self.proj= remat(nn.Conv)(features=self.embed_dim,kernel_size=self.patch_size,strides=self.patch_size)        
        self.norm = self.norm_layer()


    @nn.compact
    def __call__(self, x):
        x_shape = x.shape
        x = self.proj(x)
        x = self.norm(x)# TODO here we effectively have wrong batch norm ...
        return x




# first test patch embedding reorganize it then back to original shape will not help but should not pose problem
# then on each window we should just get simple attention with all steps ...
class Simple_window_attention(nn.Module):
    """
    basic attention based on https://theaisummer.com/einsum-attention/    
    """
    window_size: Tuple[int]
    dim:int # controls embedding 
    num_heads:int
  
    def setup(self):
        self.qkv = nn.Dense((self.dim * 3 * self.num_heads) , use_bias=False)      
        self.out_proj = nn.Dense(self.dim,  use_bias=False)      
        self.norm = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        #needed to scle dot product corectly
        head_dim = self.dim // self.num_heads
        self.scale_factor = head_dim**-0.5

    @nn.compact
    def __call__(self,_, x):
        x_shape=x.shape
        x=self.norm(x)
        qkv = self.qkv(jnp.ravel(x))
        qkv = rearrange(qkv,'(t s h) -> t s h', h= self.num_heads, s=3 )#t - token h - heads c - channels
        q,k,v = rearrange(qkv,'n split h -> split h n 1', h= self.num_heads,split=3 )
        scaled_dot_prod = einsum(q, k,'h i d , h j d -> h i j') * self.scale_factor
        # if mask is not None:
        attn= nn.activation.softmax(scaled_dot_prod,axis=- 1)
        out = einsum( attn, v,'h i j , h j d -> h i d')
        out= self.norm2(out)
        # Re-compose: merge heads with dim_head d
        out = rearrange(out, "h t d -> (t h d)")
        #Apply final linear transformation layer
        return (0,self.out_proj(out))

class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """
    img_size: Tuple[int] 
    patch_size: Tuple[int] 
    in_chans: int 
    embed_dim: int
    depths: Tuple[int] 
    num_heads: Tuple[int] 
    window_size: Tuple[int] 
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    patch_norm: bool = False
    use_checkpoint: bool = False
    spatial_dims: int = 3
    downsample="merging"
    norm_layer: Type[nn.Module] = nn.LayerNorm
    def setup(self):
        num_layers = len(self.depths)
        embed_dim_inner= np.product(list(self.window_size))
        
        self.patch_embed = PatchEmbed(
            in_channels=self.in_chans,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )        

        i=0
        length = np.product(list(self.img_size))//np.product(list(self.window_size))
        
        self.window_attention = nn.scan(
            remat(Simple_window_attention),
            in_axes=0, out_axes=0,
            variable_broadcast={'params': None},
            split_rngs={'params': False}
            ,length=length)(self.window_size,embed_dim_inner,self.num_heads[i])
        #convolutions
        self.num_features = int(self.embed_dim * 2 ** (num_layers - 1))
        self.conv_a= remat(nn.Conv)(features=self.num_features//16,kernel_size=(3,3,3),strides=(2,2,2))
        self.conv_b= remat(nn.Conv)(features=self.num_features//8,kernel_size=(3,3,3),strides=(2,2,2))
        self.conv_c= remat(nn.Conv)(features=self.num_features//4,kernel_size=(3,3,3),strides=(2,2,2))
        self.conv_d= remat(nn.Conv)(features=self.num_features//2,kernel_size=(3,3,3),strides=(2,2,2))

        self.deconv_a= remat(DeConv3x3)(features=self.num_features//4)
        self.deconv_b= remat(DeConv3x3)(features=self.num_features//8)
        self.deconv_c= remat(DeConv3x3)(features=self.num_features//16)
        self.deconv_d= remat(DeConv3x3)(features=self.num_features//16)
        self.deconv_e= remat(DeConv3x3)(features=self.num_features//16)
        self.deconv_f= remat(DeConv3x3)(features=self.num_features//16)
        self.conv_out= remat(nn.Conv)(features=1,kernel_size=(3,3,3))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            n, ch, d, h, w = x_shape
            x = einops.rearrange(x, "n c d h w -> n d h w c")
            x = nn.LayerNorm()(x)#, [ch]
            x = einops.rearrange(x, "n d h w c -> n c d h w")
        return x

    @nn.compact
    def __call__(self, x):
        # deterministic=not train
        b, c, d, h, w = x.shape
        x=einops.rearrange(x, "n c d h w -> n d h w c")
        x=self.patch_embed(x)
        x=window_partition(x, self.window_size)
        x=einops.rearrange(x, "bw t c -> bw t c")
        x=self.window_attention(0,x)[1]
        x=einops.rearrange(x, "bw (t c) -> bw t c" ,c=c)
        x=window_reverse(x, self.window_size, (b,d,h,w,c))

        x1=self.conv_a(x )
        x2=self.conv_b(x1 )
        x3=self.conv_c(x2 )
        x4=self.conv_d(x3 )
        x5=self.deconv_a(x4 )
        x6=self.deconv_b(x5+x3 )
        x7=self.deconv_c(x6+x2 )
        x8=self.deconv_d(x7+x1 )
        x9= self.conv_out(x8 )
        return einops.rearrange(x9, "n d h w c-> n c d h w")

        
        
        # x0 = self.patch_embed(x)
        # x0 = self.pos_drop(x0,deterministic=deterministic)
        # x0_out = self.proj_out(x0, normalize)      
        # # x1 = self.layers[0](x0,deterministic)
        # # x1_out = self.proj_out(x1, normalize)
        # # x2 = self.layers[1](x1,deterministic)
        # # x2_out = self.proj_out(x2, normalize)
        # # x3 = self.layers[2](x2,deterministic)
        # # x3_out = self.proj_out(x3, normalize)
        # # x4_out=self.deconv_a(x3_out,train)
        # # x4_out=einops.rearrange(x4_out, "n c d h w-> n c w h d")

        # # # print(f"x3_out {x3_out.shape} x4_out {x4_out.shape}  x2_out {x2_out.shape}")
    
        # # x5_out=self.deconv_b(x4_out+x2_out,train)
        # # x5_out=einops.rearrange(x5_out, "n c d h w-> n c h d w")
        # # x6_out=self.deconv_c(x5_out+x1_out,train )
        # # x6_out=einops.rearrange(x6_out, "n c d h w-> n c d w h")

        # # x7_out=self.deconv_d(x6_out+x0_out,train )
        # x7_out=self.deconv_d(x0_out,train )
        # x7_out=self.deconv_e(x7_out,train )
        # x7_out=self.deconv_f(x7_out,train )
        # # print(f"x3_out {x3_out.shape} x4_out {x4_out.shape}  x2_out {x2_out.shape}")
        # x8_out=einops.rearrange(x7_out, "n c d h w -> n d h w c")
        # x8_out=self.conv_out(x8_out)

        # return einops.rearrange(x8_out, "n d h w c-> n c d h w")