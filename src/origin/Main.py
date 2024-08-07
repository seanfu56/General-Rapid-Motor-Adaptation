import argparse
import warnings
import os
from Algo.SAC.SAC import sac
from Algo.TD3.TD3 import td3

import gymnasium as gym
from Algo.SAC.Model import MLPActorCritic as SacModel

from datetime import datetime



warnings.filterwarnings('ignore')

ENV              = 'Humanoid-v4'
ALGORITHM        = 'sac'
START_EPOCHS     = 10
NUM_TRAIN_EPOCHS = 3000
HIDDEN_DIM       = 256
LR               = 3e-4
NOISE_STD_MIN    = 0.1
NOISE_STD_MAX    = 0.5
NOISE_WARM_EP    = 3000
NOISE_ADD_STD    = 0.01
SEED             = 2024
BATCH_SIZE       = 256
LAYER            = 3
BUFFER_SIZE      = 1000000
PRINT            = True
SAVE             = True
SAVE_FREQ        = 100
PRINT_EPOCHS     = 100
GAMMA            = 0.99

class Hyperparameter:
    def __init__(self):
        self.env              = ENV
        self.algorithm        = ALGORITHM
        self.start_epochs     = START_EPOCHS
        self.num_train_epochs = NUM_TRAIN_EPOCHS
        self.hidden_dim       = HIDDEN_DIM
        self.lr               = LR
        self.noise_std_min    = NOISE_STD_MIN
        self.noise_std_max    = NOISE_STD_MAX
        self.noise_warm_ep    = NOISE_WARM_EP
        self.noise_add_std    = NOISE_ADD_STD
        self.seed             = SEED
        self.batch_size       = BATCH_SIZE
        self.layer            = LAYER
        self.buffer_size      = BUFFER_SIZE
        self.print            = PRINT
        self.save             = SAVE
        self.save_freq        = SAVE_FREQ
        self.print_epochs     = PRINT_EPOCHS
        self.gamma            = GAMMA

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default=ENV)
parser.add_argument('-a', '--algorithm', type=str, default=ALGORITHM)
parser.add_argument('-e', '--epoch', type=int, default=NUM_TRAIN_EPOCHS)
parser.add_argument('-hd', '--hidden_dim', type=int, default=HIDDEN_DIM)
parser.add_argument('-lr', '--learning_rate', type=float, default=LR)
parser.add_argument('-nin', '--noise_std_min', type=float, default=NOISE_STD_MIN)
parser.add_argument('-nax', '--noise_std_max', type=float, default=NOISE_STD_MAX)
parser.add_argument('-na', '--noise_add_std', type=float, default=NOISE_ADD_STD)
parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('-l', '--layer', type=int, default=LAYER)
parser.add_argument('-s', '--save', type=int, default=SAVE)

args = parser.parse_args()

hyperparameter = Hyperparameter()

hyperparameter.env              = args.env
hyperparameter.algorithm        = args.algorithm
hyperparameter.num_train_epochs = args.epoch
hyperparameter.hidden_dim       = args.hidden_dim
hyperparameter.lr               = args.learning_rate
hyperparameter.noise_std_min    = args.noise_std_min
hyperparameter.noise_std_max    = args.noise_std_max
hyperparameter.noise_add_std    = args.noise_add_std
hyperparameter.batch_size       = args.batch_size
hyperparameter.layer            = args.layer
hyperparameter.save             = args.save != 0



now = datetime.now()
format_time = now.strftime(f'%Y-%m-%d_%H-%M-{int(now.second)}')

save_dir = f'./ori_model_ckpt/{hyperparameter.env}/{format_time}'
os.makedirs(save_dir)

with open(f'{save_dir}/setting.log', 'w') as file:
    file.write('Hyperparameters: \n')
    file.write('Env: '+ str(hyperparameter.env) + '\n')
    file.write('Algorithm: '+ str(hyperparameter.algorithm) + '\n')
    file.write("Epochs: "+ str(hyperparameter.num_train_epochs) + '\n')
    file.write('Hidden_dim: '+ str(hyperparameter.hidden_dim) + '\n')
    file.write('LR: '+ str(hyperparameter.lr) + '\n')
    file.write('Noise_std_min: '+ str(hyperparameter.noise_std_min) + '\n')
    file.write('Noise_std_max: '+ str(hyperparameter.noise_std_max) + '\n')
    file.write('Noise_warm_ep: '+ str(hyperparameter.noise_warm_ep) + '\n')
    file.write('Seed: '+ str(hyperparameter.seed) + '\n')
    file.write('Batch_size: '+ str(hyperparameter.batch_size) + '\n')
    file.write('Layer: '+ str(hyperparameter.layer) + '\n')

with open(f'{save_dir}/training.log', 'w') as file:
    pass

if(hyperparameter.algorithm == 'sac'):
    sac(lambda : (gym.make(hyperparameter.env)), actor_critic=SacModel, 
        ac_kwargs=dict(hidden_sizes=[hyperparameter.hidden_dim]*hyperparameter.layer), 
        gamma=hyperparameter.gamma, seed=hyperparameter.seed, epochs=hyperparameter.num_train_epochs,
        noise_std_min=hyperparameter.noise_std_min, noise_std_max=hyperparameter.noise_std_max, noise_warm_epochs=hyperparameter.noise_warm_ep, 
        plot=True, save=hyperparameter.save, save_freq=hyperparameter.save_freq, save_dir=save_dir)


