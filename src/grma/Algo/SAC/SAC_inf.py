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
        num_test_episodes=1, max_ep_len=1000, noise_std_max=0.5, 
        plot=True, 
        pretrain_dir=None, 
        save=False,
        save_dir=None, 
        adapt_model=None, 
        length=0, 
        model_type='rnn'):

    assert(adapt_model != None)
    assert(length != 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_env = env_fn()

    adapt_model.to(device)

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

    def test_agent(type=None, id=0):

        state_list = []
        action_list = []

        o, _ = test_env.reset()
        d, ep_ret, ep_len = False, 0, 0

        pred_list = []

        noise = np.random.normal(1, 0.5, act_dim)

        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 

            state_list.append(o)

            if(ep_len < length):
                pred = np.zeros(act_dim)
            else:
                # print(np.array(state_list[-length:]).shape)
                pred_input = np.concatenate((np.array(state_list[-length:]), np.array(action_list[-length:])), axis=1)
                pred_input = pred_input.reshape(1, pred_input.shape[0], pred_input.shape[1])
                if(model_type=='cnn'):
                    pred_input = pred_input.swapaxes(1, 2)
                pred_input = torch.tensor(pred_input, dtype=torch.float).to(device)
                pred = adapt_model(pred_input)
                pred = pred.detach().cpu().numpy()
                # pred = noise - np.ones(act_dim)
                # pred = np.zeros(act_dim)
                pred_list.append(pred)

            a = get_action(np.append(o, pred), True)

            aa = a * np.array([1., 1., 1., 1., 1., 1.])
            # aa = a * noise

            action_list.append(a)

            o, r, d, _, _ = test_env.step(aa)
            ep_ret += r
            ep_len += 1

        # print(f'reward: {ep_ret:.2f}')

        return ep_ret, np.array(pred_list)


    reward_lists = []

    for i in tqdm.tqdm(range(epochs)):

        reward, pred_list = test_agent()

        # print(reward)

        # print(pred_list.shape)

        # plt.plot(pred_list[:, 0, 1])

        # plt.show()

        reward_lists.append(reward)

    print(sum(reward_lists) / len(reward_lists))




