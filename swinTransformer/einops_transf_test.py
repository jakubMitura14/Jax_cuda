from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
from einops import rearrange
import jax 
from flax.linen import partitioning as nn_partitioning
import jax

