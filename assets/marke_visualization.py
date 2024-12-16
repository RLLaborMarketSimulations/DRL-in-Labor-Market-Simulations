from .worker import WorkerState
from .firm import Firm
from .job import Job
import numpy as np
class JobMarktVis:
    def __init__(self, parameters, firms, rl_firm_index):
        self.parameters = parameters
        self.unemployed_workers = parameters['number of workers']
        self.employed_workers = parameters['number of firms'] * parameters['number of workers per firm']
        self.firms = firms
        self.v = 0
        self.m = 0
        self.w = parameters['initial b'] * parameters['z'] + (1-parameters['initial b']) * parameters['p']
        self.step_count = 0
        self.rl_firm_index = rl_firm_index

    def reset(self):
        self.unemployed_workers = self.parameters['number of workers']
        self.employed_workers = self.parameters['number of firms'] * self.parameters['number of workers per firm']
        self.v = 0
        self.m = 0
        self.w = self.parameters['initial b'] * self.parameters['z'] + (1-self.parameters['initial b']) * self.parameters['p']
        self.step_count = 0
        for firm in self.firms:
            firm.reset()


    def matching(self, u, v):
        m = int(round(self.parameters['A'] * u**self.parameters['alpha'] * v**(1-self.parameters['alpha'])))
        return np.clip(m, 0, min(u, v))

    def step(self, length, train=True):

        # input state = (u,v,m,w) --(total_unemployed, total_vacancies, total_matches, average_wage)
        # all_states = (total_unemployed, number of employees of company, total_vacancies, total_matches, match of company, average_wage, wage of company)

        state = [self.unemployed_workers, self.v, self.m, self.w]
        jobs = []
        self.v = 0
        i = 0
        for firm in self.firms:
            if self.step_count < 3:
                num_vacancies, _ = firm.fixed_action()
            else:
                num_vacancies,_ = firm.action(state)
            self.data_firm[i]['jobs'].append(num_vacancies)
            i+=1
            self.v += num_vacancies
            for _ in range(num_vacancies):
                # print(firm.wage())
                jobs.append(Job(firm, firm.wage()))

        self.data_markt['jobs'].append(self.v)
        # print('number of vacancies:', self.v)

        self.m = self.matching(self.unemployed_workers, len(jobs))
        self.data_markt['matches'].append(self.m)
        # print('number of matches:', self.m)
        # print(len(jobs))
        # print(self.m)

        all_wages = 0
        for firm in self.firms:
            num = int(round(self.parameters['lambda']*firm.num_workers))
            firm.dismiss(num)
            all_wages += firm.num_workers * firm.w
            self.unemployed_workers += num
            self.employed_workers -= num
        self.w = all_wages/self.employed_workers
        # print('average wage:', self.w)

        
        for i in range(self.m):
            n_f = min(self.parameters['n_f'], len(jobs))
            index_list = np.random.choice(len(jobs), n_f, replace=False)
            alternatives = [jobs[j].wage for j in index_list]
            best_alternative = np.argmax(alternatives)
            jobs.pop(index_list[best_alternative]).accept()

        self.unemployed_workers -= self.m
        self.employed_workers += self.m
        self.step_count += 1
        # print(self.step_count)
        self.data_markt['employed workers'].append(self.employed_workers)
        done = self.step_count >= length

        for i in range(len(self.firms)):
            self.data_firm[i]['reward'].append(self.firms[i].reward())

        for i in range(len(self.firms)):
            self.data_firm[i]['employed workers'].append(self.firms[i].num_workers)
            self.data_firm[i]['wages'].append(self.firms[i].w)

        return done
    
    def episodes(self, length):
        self.reset()
        done = False
        self.data_firm = [{'employed workers':[], 'wages':[], 'jobs':[], 'reward':[]} for i in range(len(self.firms))]
        self.data_markt = {'jobs':[], 'matches':[], 'employed workers':[]}
        while not done:
            done = self.step(length)

        print('unemployed workers:', self.unemployed_workers)
        print('employed workers:', self.employed_workers)

        for i in self.rl_firm_index:
            print('firm', i, 'reward:', self.firms[i].overall_reward, 'wage:', self.firms[i].w, 'employees:', self.firms[i].num_workers)

    def run(self, episodes, length):
        for i in range(episodes):
            print('episode:', i)
            self.episodes(length)



    # def print(self):
    #     print('Unemployed workers: ', self.unemployed_workers)
    #     print('Employed workers: ', self.employed_workers)
        # for firm in self.firms:
        #     print('Firm', firm.num_workers, 'workers')
        

        




        
