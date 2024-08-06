import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 創建示例數據
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# 定義簡單的線性回歸模型
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

# 創建圖表
plt.ion()  # 開啟交互模式
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.5)
line, = ax.plot([], [], 'r-')
losses = []

# 訓練模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # 更新圖表
    line.set_xdata(np.arange(len(losses)))
    line.set_ydata(losses)
    ax.set_xlim(0, len(losses))
    ax.set_ylim(0, max(losses) + 0.1)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # 打印當前 loss
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

plt.ioff()  # 關閉交互模式
plt.show()
