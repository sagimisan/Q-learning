import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
      super(Network,self).__init__()
      self.seed=torch.manual_seed(seed)
      self.fc1=nn.Linear(state_size,64)
      self.fc2=nn.Linear(64,64)
      self.fc3=nn.Linear(64,action_size)

    def forward(self,state):
        """
          Forward pass through the neural network.
      
          Applies two fully connected layers with ReLU activation, followed by a final output layer.
      
          Args:
              state (torch.Tensor): Input state tensor to be processed by the network.
      
          Returns:
              torch.Tensor: Output tensor representing the network's prediction or action values.
          """
        x = self.fc1(state)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        return self.fc3(x)