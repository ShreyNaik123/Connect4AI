import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        # Calculate the total input size (6 * 7)
        total_input_size = input_size[0] * input_size[1]
        
        self.linear1 = nn.Linear(total_input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      # Flatten the input tensor
      x = torch.tensor(np.array(x).flatten())
      # x = x.view(x.size(0), -1)
      x = F.relu(self.linear1(x))
      x = self.linear2(x)
      return x

  # def save
  

class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(),
                                lr = self.lr)
    self.loss = nn.MSELoss()
  
  def train_step(self, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float)
    next_state = torch.tensor(next_state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
  
    pred = self.model(state)
    target = pred.clone()
    
    # print(f"""
    #   \n
    #   ---------------------------
    #   state = {state}
    #   len = {len(state.shape)}
    #   next_state = {next_state}
    #   action={action}
    #   reward = {reward}
    #   done = {done}
    #   target = {target}
    #   --------------------\n
    #   """)
    
    if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
    
    
    for idx in range(len(done)):
      
      Q_new = reward[idx]
      if not done[idx]:
        Q_new = reward[idx] + self.gamma* torch.max(self.model(next_state[idx]))
      
      # target[idx][torch.argmax(action[idx]).item()] =Q_new
      target[torch.argmax(action[idx]).item()] = Q_new.item()

    
    loss = self.loss(target, pred)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()