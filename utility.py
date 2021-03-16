import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np


def avg_sqrt_l2_loss(X, Y):
	sqrt_l2_loss = torch.sqrt(torch.mean((X-Y)**2 + 1e-6, dim=[1,2]))
	sqrn_l2_norm = torch.sqrt(torch.mean((Y**2), dim=[1,2]))
	snr = 20 * torch.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)

	avg_sqrt_l2 = torch.mean(sqrt_l2_loss, dim=0)
	avg_snr = torch.mean(snr, dim=0)

	return avg_sqrt_l2, avg_snr

class LabelsDataset(Dataset):

	def __init__(self, datapoints,labels):

		self._datapoints = datapoints
		self._labels = labels

	def __len__(self):
		return len(self._datapoints)

	def __getitem__(self, idx):
		
		datapoint = self._datapoints[idx]
		label = self._labels[idx]

		return datapoint, label
