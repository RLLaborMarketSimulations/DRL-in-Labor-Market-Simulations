import torch
import numpy as np
# np.random.seed(0)
import pandas as pd
import torch.nn as nn
import torch.optim as optim

normalization_vector = torch.tensor([300, 300, 300,100,100,1000,1000])

class CriticNetwork(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(CriticNetwork, self).__init__()
        self.net = torch.nn.Sequential(
                torch.nn.Linear(n_observations+n_actions, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,1))
        
    def forward(self, state, action):
        state = state/normalization_vector
        x = torch.cat([state, action], 1)
        return self.net(x)
    
# Neural network for the policy
class ActorNetwork(nn.Module):
    def __init__(self, n_observations):
        super(ActorNetwork, self).__init__()
        self.net = torch.nn.Sequential(
                torch.nn.Linear(n_observations, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,32),
                torch.nn.ReLU()
                )
        
        self.num_vacancies_layer = torch.nn.Sequential(
            torch.nn.Linear(32,1),
            # torch.nn.ReLU()
            torch.nn.Sigmoid()
        )

        self.bargaining_power_layer = torch.nn.Sequential(
            torch.nn.Linear(32,1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x/normalization_vector
        x = self.net(x)
        num_vacancies = self.num_vacancies_layer(x)
        bargaining_power = self.bargaining_power_layer(x)
        return num_vacancies, bargaining_power
        # mu = self.mu(x)
        # sigma =  self.softplus(self.sigma(x))
        # return mu, sigma