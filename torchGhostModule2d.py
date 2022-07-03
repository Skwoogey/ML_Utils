import torch as t
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, Literal

class GhostModule2d(nn.Module):
	def __init__(
			self,
			in_channels: int,
			filters: int,
			kernel_size: Union[Tuple[int, int], int],
			dw_kernel_size: Union[Tuple[int, int], int, None] = None, 
			ratio: int = 2, 
			stride: Union[Tuple[int, int], int] = 1,
			dilation: Union[Tuple[int, int], int] = 1,
			padding: Union[Literal['same', 'valid'], Tuple[int, int], int] = 'valid', 
			padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
			use_bias: bool = False,
			identity_transform: nn.Module = nn.Identity(),
			ghost_transform: nn.Module = nn.Identity()
		):

		super(GhostModule2d, self).__init__()

		assert(filters % ratio == 0)
		
		dw_kernel_size = dw_kernel_size if dw_kernel_size != None else kernel_size
		conv_filters = filters // ratio
		depth_conv_filters = filters - conv_filters

		self.conv1 = nn.Conv2d(
			in_channels = in_channels, 
			out_channels = conv_filters, 
			kernel_size = kernel_size,
			stride = stride,
			padding = padding, 
			dilation = dilation, 
			groups=1, 
			bias=use_bias,
			padding_mode = padding_mode)

		self.identity_transform = identity_transform

		self.conv2 = nn.Conv2d(
			in_channels = conv_filters, 
			out_channels = depth_conv_filters, 
			kernel_size = dw_kernel_size,
			stride = 1,
			padding = 'same', 
			dilation = 1, 
			groups=conv_filters, 
			bias=use_bias,
			padding_mode = padding_mode)

		self.ghost_transform = ghost_transform

	def forward(self, x):
		x = self.conv1(x)
		x = self.identity_transform(x)

		ghost_x = self.conv2(x)
		ghost_x = self.ghost_transform(x)

		return t.cat((x, ghost_x), 1)