import numpy as np
from models.RLAgents import DDPGAgent
from models.networks import CriticNetwork, ActorNetwork
from models.replayer import Replayer


class Firm:
    def __init__(self, num_workers, bargaining_power, parameters):
        self.initial_num_workers = num_workers
        self.initial_bargaining_power = bargaining_power
        self.num_workers = num_workers
        self.bargaining_power = bargaining_power
        # self.num_vacancies = num_vacancies
        self.parameters = parameters
        self.m = 0
        self.w = parameters['initial b'] * parameters['z'] + (1-parameters['initial b']) * parameters['p']
        self.last_state = None
        self.last_action = None
        self.action_ = None
        self.state = None
        self.reward_ = None
        self.overall_reward = 0

    # def fixed_action(self, state):
    #     self.bargaining_power = 0.5
    #     self.num_vacancies = 10
    #     self.m = 0
    #     return self.num_vacancies, self.bargaining_power
    def fixed_action(self):
        self.bargaining_power = 0.5 + np.random.normal(0, 0.1)
        self.num_vacancies = int(3 + np.random.normal(0, 3))
        self.m = 0
        self.num_vacancies = np.clip(self.num_vacancies, 0, 100)
        return self.num_vacancies, self.bargaining_power
    
    def action(self, state):
        return self.fixed_action()

    def dismiss(self, n):
        self.num_workers -= n

    def reward(self):
        
        self.reward_ = ((self.parameters['p'] - self.w) * self.num_workers - self.parameters['c'] * self.num_vacancies)/10000
        # print('reward:', self.reward_)
        self.overall_reward += self.reward_
        return self.reward_
    
    def reset(self):
        self.m = 0
        self.num_workers = self.initial_num_workers
        self.bargaining_power = self.initial_bargaining_power
        self.w = self.parameters['initial b'] * self.parameters['z'] + (1-self.parameters['initial b']) * self.parameters['p']
        self.last_state = None
        self.last_action = None
        self.action_ = None
        self.state = None
        self.reward_ = None
        self.overall_reward = 0

    def store(self, state, action, reward, next_state, done):
        return
    
    def learn(self):
        return


    # def wage(self, theta):
    #     self.w = self.bargaining_power * self.parameters['z'] + (1-self.bargaining_power) * self.parameters['p'] * (1 + self.parameters['c'] * theta)
    #     return self.w
    def wage(self):
        self.w = self.bargaining_power * self.parameters['z'] + (1-self.bargaining_power) * self.parameters['p']
        return self.w

    def benifit(self, wage):
        return -self.parameters['c'] * self.num_vacancies + (self.parameters['p'] - wage) * self.num_workers
    
class TrendFollowFirm(Firm):
    def __init__(self, num_workers, bargaining_power, parameters):
        super().__init__(num_workers, bargaining_power, parameters)
        
    # def fixed_action(self):
    #     self.bargaining_power = np.clip(np.random.normal(0.5, 0.3), 0, 1)
    #     self.num_vacancies = max(int(np.random.normal(5, 10)), 0)
    #     self.m = 0
    #     return self.num_vacancies, self.bargaining_power
        
    def action(self, state):
        # input state = (u,v,m,w)
        # state = (u 0    ,e_i       1      ,v    2     ,m   3    ,m_i 4 ,w   5    ,w_i 6)
        # state = [state[0], self.num_workers, state[1], state[2], self.m, state[3], self.w]
        wage_delta = self.w - state[3]
        expected_vacancies_delta = self.m*state[1]/state[2] - self.num_vacancies
        # print('vi:' self.num_vacancies, 'm: ', state[2], 'v:')
        # print(expected_vacancies_delta)
        if wage_delta < 0 and expected_vacancies_delta < 0:
            self.bargaining_power -= self.parameters['mu_b'] * abs(wage_delta) 
        elif wage_delta > 0 and expected_vacancies_delta > 0:
            self.bargaining_power += self.parameters['mu_b'] * abs(wage_delta)
        elif wage_delta < 0 and expected_vacancies_delta > 0:
            self.num_vacancies += int(round(self.parameters['mu_v'] * abs(expected_vacancies_delta)))
        elif wage_delta > 0 and expected_vacancies_delta < 0:
            self.num_vacancies -= int(round(self.parameters['mu_v'] * abs(expected_vacancies_delta)))

        self.num_vacancies = np.clip(self.num_vacancies, 0, 100)
        self.bargaining_power = np.clip(self.bargaining_power, 0, 1)

        self.m=0
        return self.num_vacancies, self.bargaining_power
    
class IndependentRLFirm(Firm):
    def __init__(self, num_workers, bargaining_power, parameters):
        super().__init__(num_workers, bargaining_power, parameters)
        # self.RLAgent = DDPGAgent(ValueNetwork(7), PolicyNetwork(7))
        self.RLAgent = DDPGAgent(CriticNetwork(7, 2), ActorNetwork(7), Replayer(10000))

    def action(self, state):
        state = [state[0], self.num_workers, state[1], state[2], self.m, state[3], self.w]
        a = self.RLAgent.select_action(state)
        self.num_vacancies = int(round(a[0]*100))
        # print(self.num_vacancies)
        self.bargaining_power = a[1]
        # self.bargaining_power = 0.5

        # self.num_vacancies = 100
        # self.bargaining_power = 1
        self.num_vacancies = max(0, self.num_vacancies)
        self.bargaining_power = np.clip(self.bargaining_power, 0, 1)
        self.m = 0
        # print(self.num_vacancies, self.bargaining_power)
        self.last_state = self.state
        self.state = state
        self.last_action = self.action_
        self.action_ = np.array((self.num_vacancies, self.bargaining_power))
        return self.num_vacancies, self.bargaining_power
        # return self.num_vacancies, 0.5
    
    def store(self, done):
        if self.last_state is not None:
            self.RLAgent.replayer.store(self.last_state, self.last_action, self.reward_, self.state, done)

    def reset(self):
        super().reset()
        self.RLAgent.update()

    def learn(self):
        return self.RLAgent.learn()


