from dm_control import suite
import numpy as np

def get_dim(obs):
    dim = 0
    for _, array in obs.items():
        print(array)
        dim += array.shape[0]

    return dim

env = suite.load('cartpole', 'balance')

spec = env.action_spec()

# print(spec)

# print(env.observation_spec())

obs = env.observation_spec()

# print(get_dim(obs))

print(env.action_spec().minimum)

print(env.action_spec().shape)

time_step = env.reset()

total_reward = 0.

for _ in range(10):
    action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)
    total_reward += time_step.reward
    # print(time_step)

print(total_reward)
