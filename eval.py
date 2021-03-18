import os
import torch

from model import *
from utils import upsample_wav

class Solver(object):
	def __init__(self, config):

		self.config = config

		self.model_path = self.config['model_path']
		self.out_label = self.config['out_label']
		self.wav_list = self.config['wav_list']
		self.r = self.config['r']
		self.sr = self.config['sr']

	def build_model(self):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model = AudioUnet(self.num_layers)
		self.model = self.model.to(self.device)
		print('model created')

	def load_model(self):
		checkpoint = torch.load(self.model_path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.eval()

	def eval(self):
		self.build_model()
		self.load_model()

		if self.out_label:
			with open(self.wav_list) as f:
				for line in f:
					try:
						print(line.strip())
						upsample_wav(line.strip(), self.config, self.model)
					except EOFError:
						print('WARNING: Error reading file:', line.strip())

