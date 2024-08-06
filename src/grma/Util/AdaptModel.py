import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, length):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear((state_dim + action_dim) * length, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, state_dim, action_dim, length, hidden_dim=16):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM((state_dim + action_dim), hidden_dim, num_layers=1, batch_first=False)
        self.fc1 = nn.Linear(hidden_dim * length, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, action_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        # 線性轉換至 output
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.tanh(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, state_dim, action_dim, length, kernel_size=5, out_channels=8):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=(state_dim + action_dim), out_channels=32, 
                            kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=out_channels
                            , kernel_size=kernel_size, stride=1)
        self.fc1 = nn.Linear(out_channels * (length-2*kernel_size+2), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.tanh(x)
        return x