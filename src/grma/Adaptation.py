from Util.AdaptModel import RNN, CNN, MLP

import numpy as np
import torch
import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import os
import tqdm
import gymnasium as gym

from torchinfo import summary

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

class NpDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        return image, label

EPOCH = 1000

ENV = 'HalfCheetah-v4'
MODEL = 'mlp'


STATE_DIM = 17
ACTION_DIM = 6

LENGTH = 30
LR = 3e-3

dir = f'./traj/{ENV}'

BATCH_SIZE = 4096

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# state = np.random.normal(0, 1, (TEST_EPOCH, TIME_STEP, STATE_DIM))
# action = np.random.normal(0, 1, (TEST_EPOCH, TIME_STEP, ACTION_DIM))
# noise = np.random.normal(0, 1, (TEST_EPOCH, ACTION_DIM))

with open(f'{dir}/state.npy', 'rb') as f:
    state = np.load(f)

with open(f'{dir}/action.npy', 'rb') as f:
    action = np.load(f)

with open(f'{dir}/noise.npy', 'rb') as f:
    noise = np.load(f)

TEST_EPOCH = state.shape[0]
TIME_STEP = state.shape[1]

noise = np.tile(noise[:, np.newaxis, :], (1, TIME_STEP//LENGTH, 1))


noise = noise.reshape(TEST_EPOCH * TIME_STEP//LENGTH, ACTION_DIM)
noise = torch.tensor(noise, dtype=torch.float).to(device)

# TODO
if(MODEL == 'cnn'):
    model = CNN(STATE_DIM, ACTION_DIM, LENGTH).to(device)
    modelinfo = summary(model, (1, STATE_DIM+ACTION_DIM, LENGTH), verbose=0)
elif(MODEL == 'rnn'):
    model = RNN(STATE_DIM, ACTION_DIM, LENGTH).to(device)
    modelinfo = summary(model, (1, LENGTH, STATE_DIM+ACTION_DIM), verbose=0)
elif(MODEL == 'mlp'):
    model = MLP(STATE_DIM, ACTION_DIM, LENGTH).to(device)
    modelinfo = summary(model, (1, (STATE_DIM + ACTION_DIM) * LENGTH), verbose=0)

with open(f'./adapt_model/{ENV}/{MODEL}.log', 'w') as f:
    f.write(str(modelinfo) + '\n')

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

state = state.reshape((TEST_EPOCH, TIME_STEP//LENGTH, LENGTH, STATE_DIM))
action = action.reshape((TEST_EPOCH, TIME_STEP//LENGTH, LENGTH, ACTION_DIM))
data = np.concatenate((state, action), axis=3)


criterion = nn.MSELoss()

x = data.reshape(TEST_EPOCH * TIME_STEP//LENGTH, LENGTH, STATE_DIM + ACTION_DIM)
if(MODEL == 'cnn'):
    x = np.swapaxes(x, 1, 2)
elif(MODEL == 'rnn'):
    # x = x[0: 2000, :, :]
    # noise = noise[0: 2000, :]
    pass
elif(MODEL == 'mlp'):
    x = x.reshape(TEST_EPOCH * TIME_STEP//LENGTH, LENGTH * (STATE_DIM + ACTION_DIM))


x = torch.tensor(x, dtype=torch.float).to(device)

train_data, valid_data, train_label, valid_label = train_test_split(x, noise, test_size=0.025, random_state=42)

train_dataset = NpDataset(train_data, train_label)
valid_dataset = NpDataset(valid_data, valid_label)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

min_loss = np.inf

plt.ion()
fig, ax = plt.subplots()
train_line, = ax.plot([], [], 'r-', label='training loss')
valid_line, = ax.plot([], [], 'b-', label='validation loss')

ax.legend()

train_loss_list = []
valid_loss_list = []

for epoch in tqdm.tqdm(range(EPOCH), ncols=100):

    for x, label in train_dataloader:

        optimizer.zero_grad()

        y = model(x)

        loss = criterion(y, label)

        loss.backward()

        optimizer.step()

    for xx, ll in valid_dataloader:

        with torch.no_grad():

            yy = model(xx)

            v_loss = criterion(yy, ll)

    if(v_loss.item() < min_loss):
        min_loss = v_loss.item()
        print(f'training_loss: {loss.item():.5f}, valid_loss: {v_loss.item():.5f}, lr: {optimizer.param_groups[0]['lr']:.6f}, ***min_loss***')
    else: 
        print(f'training_loss: {loss.item():.5f}, valid_loss: {v_loss.item():.5f}, lr: {optimizer.param_groups[0]['lr']:.6f}') 

    scheduler.step()

    train_loss_list.append(loss.item())
    valid_loss_list.append(v_loss.item())

    train_line.set_xdata(range(epoch+1))
    train_line.set_ydata(train_loss_list)
    valid_line.set_xdata(range(epoch+1))
    valid_line.set_ydata(valid_loss_list)
    ax.set_xlim(0, epoch+1)
    ax.set_ylim(0, max(max(train_loss_list), max(valid_loss_list)))

    fig.canvas.draw()
    fig.canvas.flush_events()

dir = f'./adapt_model/{ENV}'

os.makedirs(dir, exist_ok=True)

torch.save(model.state_dict(), f'{dir}/{MODEL}.pth')

plt.savefig(f'{dir}/{MODEL}.png')

plt.ioff()