import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


# n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
# n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]

n_filters = [  128,  256,  512, 512]
n_filters_up = [  128,  128,  256, 512]
n_filtersizes = [65, 33, 17,  9]


def SubPixel1d(tensor, r): #(b,r,w)
    ps = nn.PixelShuffle(r)
    tensor = torch.unsqueeze(tensor, -1) #(b,r,w,1)
    tensor = ps(tensor)
    #print(tensor.shape) #(b,1,w*r,r)
    tensor = torch.mean(tensor, -1)
    #print(tensor.shape) #(b,1,w*r)
    return tensor

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

class Up1D(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride, padding):
		super(Up1D, self).__init__()

		self.c1 = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel, stride=2, padding=int(kernel/2 ) )
		nn.init.orthogonal_(self.c1.weight)
		self.drop = nn.Dropout(p=0.5)

	def SubPixel1d(tensor, r): #(b,r,w)
		ps = nn.PixelShuffle(r)
		tensor = torch.unsqueeze(tensor, -1) #(b,r,w,1)
		tensor = ps(tensor)
		#print(tensor.shape) #(b,1,w*r,r)
		tensor = torch.mean(tensor, -1)
		#print(tensor.shape) #(b,1,w*r)
		return tensor

	def forward(self, x):
		# print('x1',x.shape)
		x1 = self.c1(x)
		# print('c1',x1.shape)
		x1 = self.drop(x1)
		# print('drop',x1.shape)
		x1 = F.relu(x1)
		# print('relu',x1.shape)
		# x1 = SubPixel1d(x, r=2)
		# print('pixel',x1.shape)
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

class AudioUnet(nn.Module):
	def __init__(self, num_layers):
		super(AudioUnet, self).__init__()
		self.downsample = nn.ModuleList([])
		in_channels = 128
		for l, nf, fs in zip(range(num_layers), n_filters, n_filtersizes):
			# print(l,nf,fs)
			self.downsample.append(Down1D(in_channels, nf, fs, 2, fs/2))
			in_channels = nf

		# self.bottleneck = Bottleneck(in_channels, 256, n_filtersizes[-1], 2, int(n_filtersizes[-1]/2))
		
		self.upsample = nn.ModuleList([])
		x = 1
		for l, nf, fs in list(reversed(list(zip(range(num_layers), n_filters_up, n_filtersizes)))):
			# x = x+1
			# print(x)
			self.upsample.append(Up1D(in_channels, nf, fs, 1, fs/2))
			# print(self.upsample[num_layers-1-l])
			in_channels = nf*2

		self.final = nn.Conv1d(in_channels, 128, 9, stride=2, padding=5)
		nn.init.normal_(self.final.weight)

	def forward(self, x):
		print(x.shape)
		num_layers = 4
		down_outs = [x]
		for i in range(num_layers):
			down_outs.append(self.downsample[i](down_outs[i]))
			# print('1',down_outs[i].shape)
		x1 = down_outs[-1]
		# x1 = self.bottleneck(down_outs[-1])
		for i in range(num_layers):
			# print(i)
			x1 = self.upsample[i](x1)
			# print('2',x1.shape)
			# print('3',down_outs[num_layers-i-1].shape)
			x1 = torch.cat([x1, down_outs[num_layers-i-1]],axis=1) #concat axis =-1 for tf
		x1 = self.final(x1)
		print('1',x1.shape)
		# x1 = SubPixel1d(x1, r=2)
		x1 = x1 + x

		return x1

