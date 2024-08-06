from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam

import gymnasium as gym

import time
import Algo.SAC.Model as Model 

import matplotlib.pyplot as plt

import tqdm
    

def sac(env_fn, actor_critic=Model.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=1000, epochs=100, gamma=0.99, 
        alpha=0.2, 
        num_test_episodes=1, max_ep_len=120, noise_std_max=0.5, 
        plot=True, 
        pretrain_dir=None, 
        save=False,
        save_dir=None):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_env = env_fn()

    act_dim = test_env.action_space.shape[0]
    obs_dim = test_env.observation_space.shape[0] + act_dim
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = test_env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim, act_limit, **ac_kwargs).to(device)

    ac.pi.load_state_dict(torch.load(pretrain_dir))

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), 
                    deterministic)

    def test_agent():

        reward_list = []
        state_list = []
        action_list = []

        o, _ = test_env.reset()
        d, ep_ret, ep_len = False, 0, 0

        noise = np.random.normal(0, noise_std_max, act_dim)
        noise = np.clip(noise, -1, 1)

        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            
            a = get_action(np.append(o, noise), True)

            state_list.append(o)
            action_list.append(a)

            o, r, d, _, _ = test_env.step(a * (np.ones(act_dim) + noise))
            ep_ret += r
            ep_len += 1

        # print(f'reward: {ep_ret:.2f}')

        return state_list, action_list, noise, ep_ret

    state_lists  = []
    action_lists = []
    noise_lists = []

    for i in tqdm.tqdm(range(epochs)):

        state_list, action_list, noise, reward = test_agent()

        # print(len(state_list))
        if(len(state_list) == max_ep_len):
            state_lists.append(state_list)
            action_lists.append(action_list)
            noise_lists.append(noise)

    state_lists = np.array(state_lists)
    action_lists = np.array(action_lists)
    noise_lists = np.array(noise_lists)

    print(state_lists.shape)
    print(action_lists.shape)
    print(noise_lists.shape)

    if(save):
        with open(f'{save_dir}/state.npy', 'wb') as f:
            np.save(f, state_lists)

        with open(f'{save_dir}/action.npy', 'wb') as f:
            np.save(f, action_lists)

        with open(f'{save_dir}/noise.npy', 'wb') as f:
            np.save(f, noise_lists)


