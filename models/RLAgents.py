import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import copy

class OrnsteinUhlenbeckProcess:
    def __init__(self, x0):
        self.x = x0

    def __call__(self, mu=0., sigma=1., theta=.15, dt=.01):
        n = np.random.normal(size=self.x.shape)
        self.x += (theta * (mu - self.x) * dt + sigma * np.sqrt(dt) * n)
        return self.x

class DDPGAgent():
    def __init__(self, critic_net, actor_net, replayer):
        self.critic_net = critic_net
        self.actor_net = actor_net
        self.replayer = replayer
        self.target_critic_net = copy.deepcopy(critic_net)
        self.target_actor_net = copy.deepcopy(actor_net)
        self.discount = 0.99

        self.actor_optim = optim.Adam(actor_net.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(critic_net.parameters(), lr=1e-3)

        self.critic_loss = nn.MSELoss()
        self.noise = OrnsteinUhlenbeckProcess(np.zeros((2,)))


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            num_vacancies, bargaining_power = self.actor_net(state)
        noise = self.noise(sigma = 0.01)
        
        return num_vacancies.item()+noise[0], bargaining_power.item()+noise[1]
    
    def update_net(self, target_net, evaluate_net, learning_rate=0.005):
        for target_param, evaluate_param in zip(
                target_net.parameters(), evaluate_net.parameters()):
            target_param.data.copy_(learning_rate * evaluate_param.data
                    + (1 - learning_rate) * target_param.data)
    
    def learn(self):
        if self.replayer.count < 128:
            return 0, 0
        states, actions, rewards, next_states, dones = self.replayer.sample(128)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions*np.array([0.001,1]), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        next_action_tensor = self.target_actor_net(next_states)
        # next_action_tensor = torch.tensor(next_action_tensor, dtype=torch.float32)
        next_action_tensor = torch.cat(next_action_tensor, 1)
        noise_tensor = 0.1*torch.randn_like(next_action_tensor, dtype=torch.float32)+1
        noisy_next_action_tensor = next_action_tensor*noise_tensor
        next_q_tensor = self.target_critic_net(next_states, noisy_next_action_tensor)
        critic_target = rewards + self.discount * (1 - dones) * next_q_tensor.squeeze()
        critic_target = critic_target.detach()

        critic_pred_tensor = self.critic_net(states, actions).squeeze()
        critic_loss = self.critic_loss(critic_pred_tensor, critic_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        pred_action_tensor = self.actor_net(states)
        pred_action_tensor = torch.cat(pred_action_tensor, 1)
        critic_pred_tensor = self.critic_net(states, pred_action_tensor)
        actor_loss = -critic_pred_tensor.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
                
        return critic_loss.item(), actor_loss.item()
    
    def update(self):
        self.update_net(self.target_critic_net, self.critic_net)
        self.update_net(self.target_actor_net, self.actor_net)


