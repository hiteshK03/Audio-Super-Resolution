import os
import time
import h5py
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import *
from utility import *
from utils import load_h5 

import librosa

default_opt = {'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999, 'num_layers': 4, 'batch_size': 128}

class Solver(object):
	def __init__(self, config):

		self.config = config

		self.train_path = self.config['train_path']
		self.eval_path = self.config['eval_path']

		self.epoch = self.config['epoch']
		self.batch_size = self.config['batch_size']
		self.log_dir = self.config['log_dir']
		self.num_layers = self.config['num_layers']
		self.alg = self.config['alg']
		self.lr = self.config['lr']
		self.betas = (self.config['b1'], self.config['b2'])
		
		self.model_save_dir = self.config['model_save_dir']
		self.model_save_step = self.config['model_save_step']

	def build_model(self):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model = AudioUnet(self.num_layers)

		#Data Parallel
		# if torch.cuda.device_count() > 1:
		# 	model = nn.DataParallel(model)
		self.model = self.model.to(self.device)

		if self.alg == "adam":
			self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, betas=self.betas)
		else:
			raise ValueError('Invalid Optimizer: ' + self.alg)

		print('model created')

	def print_network(self):
		"""Print out the network information"""
		num_params = 0
		for p in self.model.parameters():
			num_params += p.numel()
		print(self.model)
		print("The number of parameters: {}".format(num_params))

	def build_tensorboard(self):
		"""Build a tensorboard SummaryWriter"""
		from torch.utils.tensorboard import SummaryWriter
		self.writer = SummaryWriter(self.log_dir)

	def load_dataset(self):
		"""Load the dataset"""
		
		X_train, Y_train = load_h5(self.train_path)
		X_val, Y_val = load_h5(self.eval_path)

		# determine super-resolution level
		n_dim, n_chan = Y_train[0].shape
		self.r = Y_train[0].shape[1] / X_train[0].shape[1]
		assert n_chan == 1

		self.train_dataset = LabelsDataset(X_train, Y_train)
		self.eval_dataset = LabelsDataset(X_val, Y_val)
		print('dataset loaded')

	def load_model(self, resume_training, path):
		# model_path = os.path.join(self.model_save_dir, 'model.tar')
		checkpoint = torch.load(path)
		if resume_training:
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.curr_epoch = checkpoint['epoch']
		else: # for testing purposes (currently not used)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.model.eval()

	def val(self, dataset, n_batch=128):
		"""Error Evaluation loops"""
		batch_iterator = DataLoader(dataset, n_batch, shuffle=True, num_workers=4)

		l2_loss, l2_snr = 0, 0
		tot_l2_loss, tot_snr = 0.0, 0.0
		self.model.eval()
		for X, Y in batch_iterator:
			X, Y = X.to(self.device), Y.to(self.device) 
			output = self.model(X)
			l2_loss, l2_snr = avg_sqrt_l2_loss(Y, output)
			tot_l2_loss += l2_loss.item()
			tot_snr += l2_snr.item()
		return tot_l2_loss / len(batch_iterator), tot_snr / len(batch_iterator)

	def train(self):

		self.build_model()
		self.print_network()
		self.build_tensorboard()
		self.load_dataset()

		model_path = os.path.join(self.model_save_dir, 'model.tar')
		if os.path.exists(model_path):
			self.load_model(resume_training=True, path=model_path)
		else:
			self.curr_epoch = 0

		data_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)
		print('data loaded')

		#train Loops

		for self.curr_epoch in range(self.epoch):
			start_time = time.time()
			i = 0
			tot_l2_loss = 0.0
			tot_l2_snr = 0.0
			print('Epoch [{}/{}]'.format(self.curr_epoch+1,self.epoch))
			self.model.train()
			for X, Y in data_loader:
				X, Y = X.to(self.device), Y.to(self.device)
				output = self.model(X)
				tr_l2_loss, tr_l2_snr = avg_sqrt_l2_loss(Y, output)
				self.optimizer.zero_grad()
				tr_l2_loss.backward()
				self.optimizer.step()
				
				tot_l2_loss += tr_l2_loss.item()
				tot_l2_snr += tr_l2_snr.item()

				i+=1
				if i%100 == 0:
					print("Batch number: {:03d}, Training: Loss: {:.4f}, SNR: {:.4f}".format(i, tot_l2_loss/(i), tot_l2_snr/(i)))

			end_time = time.time()
			avg_l2_loss = tot_l2_loss/len(data_loader)
			avg_snr_loss = tot_l2_snr/len(data_loader)
			va_l2_loss, va_l2_snr = self.val(self.eval_dataset, n_batch=self.batch_size)

			print("Epoch {} of {} took {:.3f}s ({} minibatches)".format(self.curr_epoch+1, self.epoch, end_time-start_time, len(data_loader)))
			print("Training l2_loss/segsnr:\t\t{:.6f}\t{:.6f}".format(avg_l2_loss, avg_snr_loss))
			print("Validation l2_loss/segsnr:\t\t{:.6f}\t{:.6f}".format(va_l2_loss, va_l2_snr))
			
			#Add Scalar to Summary Writer
			self.writer.add_scalar('tr_l2_loss', avg_l2_loss, self.curr_epoch)
			self.writer.add_scalar('tr_l2_snr', avg_snr_loss, self.curr_epoch)
			self.writer.add_scalar('va_l2_loss', va_l2_loss, self.curr_epoch)
			self.writer.add_scalar('va_l2_snr', va_l2_snr, self.curr_epoch)

			#checkpoint the model
			if (self.curr_epoch+1) % self.model_save_step == 0:
				model_path = os.path.join(self.model_save_dir, 'model.tar')
				torch.save({
					'epoch':self.curr_epoch,
					'model_state_dict': self.model.state_dict(),
					'optimizer_state_dict': self.optimizer.state_dict(),
					}, model_path)
				print('Saved Model checkpoints into {}'.format(self.model_save_dir))

		torch.save(self.model.state_dict(), './AudioUnet.pt')
