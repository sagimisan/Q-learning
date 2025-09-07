import torch
from Lunar_Landing import Network
import torch.optim as optim
from Hyperparameters import learning_rate , replay_buffer_size,minibatch_size, discount_factor
from ReplayMemory import ReplayMemory
import random
import numpy as np
import torch.nn.functional as F

class Agent():

    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size=state_size
        self.action_size=action_size

        self.local_qnetwork=Network( state_size, action_size).to(self.device)
        self.local_target_qnetwork=Network( state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(),lr = learning_rate)

        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state ,action, reward, next_state,done):
        self.memory.push((state ,action, reward, next_state,done))
        self.t_step =(self.t_step+1)%4
        if self.t_step==0:
            if len(self.memory.memory)> minibatch_size:
                experiences= self.memory.sample(100)
                self.learn(experiences,discount_factor)

    def act(self, state,epsilon=0.):
        state= torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values= self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self,experiences,gamma):
        states,next_states,actions,rewards,dones=experiences
        next_Q_targets=self.local_target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets=rewards+gamma*next_Q_targets*(1-dones)
        Q_expected=self.local_qnetwork(states).gather(1,actions)
        loss=F.mse_loss(Q_expected,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork,self.local_target_qnetwork,1e-3)

    
    def soft_update(self,local_model,target_model,implentation_parameter):
        for target_param,local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(implentation_parameter*local_param.data+(1-implentation_parameter)*target_param.data)