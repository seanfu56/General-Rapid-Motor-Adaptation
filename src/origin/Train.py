import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import os
import time
import shutil
from Algo.TD3.TD3 import TD3
from collections import deque
from tqdm import tqdm
from Util.Replay import Replay

def save(agent, directory, filename):
    torch.save(agent.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(agent.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    torch.save(agent.actor_target.state_dict(), '%s/%s_actor_t.pth' % (directory, filename))
    torch.save(agent.critic_target.state_dict(), '%s/%s_critic_t.pth' % (directory, filename))   
    torch.save(agent.actor_optimizer.state_dict(), '%s/%s_actor_optim.pth' % (directory, filename))
    torch.save(agent.critic_optimizer.state_dict(), '%s/%s_critic_optim.pth' % (directory, filename))

def sign(x):
    if(x >= 0):
        return ''

    else:
        return 'm'
    
P_START_EPOCH = 10
P_NUM_TRAIN_EPOCHS = 3000
P_PRETRAIN = False
P_PATH = ""
P_LR = 3e-4
P_STD_NOISE = 5e-2
P_SEED = 12345
P_PRINT = True
P_PRINT_EPOCH = 100
P_SAVE = 100
P_LAYER = 3

START_EPOCH = P_START_EPOCH
NUM_TRAIN_EPOCHS = P_NUM_TRAIN_EPOCHS
PRETRAIN = P_PRETRAIN
PATH = P_PATH
LR = P_LR
STD_NOISE = P_STD_NOISE
SEED = P_SEED
PRINT = P_PRINT
SAVE = P_SAVE
LAYER = P_LAYER
PRINT_EPOCH = P_PRINT_EPOCH

C_ENV = "HalfCheetah-v4"

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epoch', type=int, default=0, help="The number of episodes in training.")
parser.add_argument('-lr', '--learning_rate', type=float, default=0, help="Learning rate")
parser.add_argument('-p', '--print', type=int, default=-1, help="Print information or not. 0 for not print, 1 for print.")
parser.add_argument('-l', '--layer', type=int, default=0, help="Layer of neural network")

args = parser.parse_args()

print(args)

if(args.epoch != 0):
    NUM_TRAIN_EPOCHS = args.epoch

if(args.learning_rate != 0):
    LR = args.learning_rate

if(args.print != -1):
    if(args.print == 0):
        PRINT = False
    elif(args.print == 1):
        PRINT = True
    else:
        PRINT = PRINT

if(args.layer != 0):
    LAYER = args.layer

ENV = C_ENV

env = gym.make(ENV)
torch.manual_seed(SEED)
np.random.seed(SEED)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MAX_ACTION = float(env.action_space.high[0])
THRESHOLD = env.spec.reward_threshold
MAX_STEPS = env.spec.max_episode_steps

agent = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION, NUM_TRAIN_EPOCHS, LR, PRETRAIN, LAYER)

if(PRINT):
    print(agent.actor)
    print(agent.critic)

scores_deque = deque(maxlen=100)
scores_array = []
avg_scores_array = []
time_start = time.time()

replay = Replay()

total_timesteps = 0

low = env.action_space.low
high = env.action_space.high

lr_list = []

for epoch in tqdm(range(1, NUM_TRAIN_EPOCHS + 1), ncols=80):
     
    timestep = 0
    total_reward = 0

    state, _ = env.reset()

    done = False

    for i in range(MAX_STEPS):

        if(epoch <= P_START_EPOCH and PRETRAIN == False):
            action = env.action_space.sample()

        else:
            action = agent.select_action(np.array(state))

            if STD_NOISE != 0:
                action_noise = np.random.normal(0, STD_NOISE, size=ACTION_DIM)
                action = (action + action_noise).clip(low, high)

        action_input = action

        new_state, reward, done, _, _ = env.step(action_input)
        done_bool = 0 if timestep + 1 == MAX_STEPS else float(done)

        total_reward += reward

        replay.add((state, new_state, action, reward, done_bool))

        state = new_state

        timestep += 1
        total_timesteps += 1

        if done:
            break
    
    scores_deque.append(total_reward)
    scores_array.append(total_reward)

    avg_score = np.mean(scores_deque)
    avg_scores_array.append(avg_score)

    max_score = np.max(scores_deque)

    lr = agent.train(replay, timestep, policy_noise=0.2, noise_clip=0.5)

    s = (int)(time.time() - time_start)

    if(epoch % 10 == 0):
        print(f'Ep. {epoch}, Avg. {avg_score:.2f}, Max. {max_score:.2f}, LR. {lr:.6f}, Time: {s//3600:02}:{s%3600//60:02}:{s%60:02}') 


state = env.reset()
