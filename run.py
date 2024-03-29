import sys
sys.path.append('path_to_main_folder')

import os
import matplotlib
matplotlib.use('Agg')

import argparse

from model import *
from train import Trainer
from eval import Solver

# ----------------------------------------------------------------------------

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train

  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(func=train)
 
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  train_parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train')
  train_parser.add_argument('--batch-size', type=int, default=128,
    help='training batch size')
  train_parser.add_argument('--logdir', default='./logs/',
    help='folder where logs will be stored')
  train_parser.add_argument('--layers', default=4, type=int,
    help='number of layers in each of the D and U halves of the network')
  train_parser.add_argument('--alg', default='adam',
    help='optimization algorithm')
  train_parser.add_argument('--lr', default=1e-3, type=float,
    help='learning rate')
  train_parser.add_argument('--b1', default=0.99, type=float,
    help='beta1 for adam')
  train_parser.add_argument('--b2', default=0.999, type=float,
    help='beta2 for adam')
  train_parser.add_argument('--save_dir', default='./output/',
    help='Directory to save Model checkpoints')
  train_parser.add_argument('--save_step', default=1, type=int,
    help='epochs after which to save the model')


  # eval

  eval_parser = subparsers.add_parser('eval')
  eval_parser.set_defaults(func=eval)

  eval_parser.add_argument('--logname', default='./output/model.tar',
    help='path to training checkpoint')
  eval_parser.add_argument('--out_label', default='',
    help='append label to output samples')
  eval_parser.add_argument('--wav_file_list', 
    help='list of audio files for evaluation')
  eval_parser.add_argument('--r', help='upscaling factor', type=int)
  eval_parser.add_argument('--sr', help='high-res sampling rate', 
                                   type=int, default=16000)
  
  return parser

# ----------------------------------------------------------------------------

def train(args):
  # train model
  config = { 'train_path':args.train, 'eval_path':args.val, 'epoch':args.epochs ,'alg' : args.alg, 'lr' : args.lr, 'b1' : args.b1, 'b2':args.b2,
                   'batch_size': args.batch_size, 'num_layers': args.layers,'log_dir': args.logdir, 'model_save_dir':args.save_dir, 'model_save_step':args.save_step}
  sol = Trainer(config)
  sol.train()

def eval(args):
  # eval model

  if os.path.exists(args.logname):
    config = {'model_path':args.logname, 'out_label':args.out_label, 'wav_list':args.wav_file_list ,'r': args.r, 'sr':args.sr}
    sol = Solver(config)
    sol.eval()
  else:
    print("No ckpt file")

def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()
