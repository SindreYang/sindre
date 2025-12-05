import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
from einops import rearrange
from sindre.ai.layers import MLP,attention,QKNorm

