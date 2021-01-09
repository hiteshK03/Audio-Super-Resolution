import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]

# n_filters = [  128,  256,  512, 512]
# n_filters_up = [  128,  128,  256, 512]
# n_filtersizes = [65, 33, 17,  9]

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        # print(x.shape)

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        # print(x.shape)

        return x

# def SubPixel1d(tensor, r): #(b,r,w)
#     ps = nn.PixelShuffle(r)
#     print(tensor.shape)
#     tensor = torch.unsqueeze(tensor, -1) #(b,r,w,1)
#     print(tensor.shape)
#     tensor = ps(tensor)
#     print(tensor.shape) #(b,1,w*r,r)
#     tensor = torch.mean(tensor, -1)
#     print(tensor.shape) #(b,1,w*r)
#     return tensor

class Down1D(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride, padding):
		super(Down1D, self).__init__()

		self.c1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=2, padding=int(kernel/2 )) 
		nn.init.orthogonal_(self.c1.weight)

	def forward(self, x):
		x1 = self.c1(x)
		x1 = F.leaky_relu(x1, negative_slope=0.2)
		return x1

class Bottleneck(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride, padding):
		super(Bottleneck, self).__init__()

		self.c1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=2, padding=int(kernel/2 ) )
		nn.init.orthogonal_(self.c1.weight)
		self.drop = nn.Dropout(p=0.5)


	def forward(self, x):
		x1 = self.c1(x)
		x1 = self.drop(x1)
		x1 = F.leaky_relu(x1, negative_slope=0.2)
		return x1

class Up1D(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride, padding):
		super(Up1D, self).__init__()

		self.c1 = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel, stride=2, padding=int(kernel/2 ) )
		nn.init.orthogonal_(self.c1.weight)
		self.drop = nn.Dropout(p=0.5)
		self.pix = PixelShuffle1D(2)

	def forward(self, x):
		# print('x1',x.shape)
		x1 = self.c1(x)
		# print('c1',x1.shape)
		x1 = self.drop(x1)
		# print('drop',x1.shape)
		x1 = F.relu(x1)
		# print('relu',x1.shape)
		x1 = self.pix(x1)
		# print('pixel',x1.shape)
		return x1


class AudioUnet(nn.Module):
	def __init__(self, num_layers):
		super(AudioUnet, self).__init__()
		self.num_layers = num_layers
		self.downsample = nn.ModuleList([])
		in_channels = 128
		for l, nf, fs in zip(range(self.num_layers), n_filters, n_filtersizes):
			# print(l,nf,fs)
			self.downsample.append(Down1D(in_channels, nf, fs, 2, fs/2))
			in_channels = nf

		self.bottleneck = Bottleneck(in_channels, n_filters[-1], n_filtersizes[-1], 2, int(n_filtersizes[-1]/2))
		
		in_channels = n_filters[-1]
		self.upsample = nn.ModuleList([])
		# x = 1
		for l, nf, fs in list(reversed(list(zip(range(self.num_layers), n_filters, n_filtersizes)))):
			# x = x+1
			# print('a',nf)
			self.upsample.append(Up1D(in_channels, 2*nf, fs, 1, fs/2))
			# print(self.upsample[num_layers-1-l])
			in_channels = nf

		self.final = nn.Conv1d(in_channels, 2, 9, stride=2, padding=5)
		self.pix = PixelShuffle1D(2)
		nn.init.normal_(self.final.weight)

	def forward(self, x):
		print(x.shape)
		# num_layers = 4
		down_outs = [x]
		for i in range(self.num_layers):
			down_outs.append(self.downsample[i](down_outs[i]))
			# print('1',down_outs[i+1].shape)
		# x1 = down_outs[-1]
		x1 = self.bottleneck(down_outs[-1])
		for i in range(self.num_layers):
			# print(x1.shape)
			x1 = self.upsample[i](x1)
			# print('2',x1.shape)
			# print('3',down_outs[self.num_layers-i].shape)
			x1 = torch.cat([x1, down_outs[self.num_layers-i]],axis=2) #concat axis =-1 for tf
			# print('passed')
		x1 = self.final(x1)
		# print('1',x1.shape)
		x1 = self.pix(x1)
		x1 = x1 + x

		return x1