import numpy as np
import h5py
import librosa
import torch
from torch.utils.data import Dataset

from scipy.signal import decimate

from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------

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

def avg_sqrt_l2_loss(X, Y):
	sqrt_l2_loss = torch.sqrt(torch.mean((X-Y)**2 + 1e-6, dim=[1,2]))
	sqrn_l2_norm = torch.sqrt(torch.mean((Y**2), dim=[1,2]))
	snr = 20 * torch.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)

	avg_sqrt_l2 = torch.mean(sqrt_l2_loss, dim=0)
	avg_snr = torch.mean(snr, dim=0)

	return avg_sqrt_l2, avg_snr

def load_h5(h5_path):
  # load training data
  with h5py.File(h5_path, 'r') as hf:
    # print ('List of arrays in input file:')
    # print(hf.keys())
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    # print ('Shape of X:', X.shape)
    # print ('Shape of Y:', Y.shape)
    print('data loaded from .h5')
  return X, Y

def upsample_wav(wav, params, model):
  # load signal
  x_hr, fs = librosa.load(wav, sr=params['sr'])

  # downscale signal
  # x_lr = np.array(x_hr[0::args.r])
  x_lr = decimate(x_hr, params['r'])
  # x_lr = decimate(x_hr, args.r, ftype='fir', zero_phase=True)
  # x_lr = downsample_bt(x_hr, args.r)

  # upscale the low-res version
  P = model.forward(x_lr.reshape((1,len(x_lr),1)))
  x_pr = P.flatten()

  # crop so that it works with scaling ratio
  x_hr = x_hr[:len(x_pr)]
  x_lr = x_lr[:len(x_pr)]

  # save the file
  outname = wav + '.' + params.out_label
  librosa.output.write_wav(outname + '.hr.wav', x_hr, fs)  
  librosa.output.write_wav(outname + '.lr.wav', x_lr, fs / params['r'])  
  librosa.output.write_wav(outname + '.pr.wav', x_pr, fs)  

  # save the spectrum
  S = get_spectrum(x_pr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.pr.png')
  S = get_spectrum(x_hr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.hr.png')
  S = get_spectrum(x_lr, n_fft=2048/params['r'])
  save_spectrum(S, outfile=outname + '.lr.png')

# ----------------------------------------------------------------------------

def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  S = np.log1p(np.abs(S))
  return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
  plt.imshow(S.T, aspect=10)
  # plt.xlim([0,lim])
  plt.tight_layout()
  plt.savefig(outfile)