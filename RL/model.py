import torch
from torch.distributions import normal,bernoulli
import math
torch.manual_seed(14)

class Bandit(object):
    def __init__(self,num_actions):
        self.num_actions = num_actions
        self.true_values = torch.randn((self.num_actions))
        self.est_values = torch.zeros((num_actions,))
        self.action_sample_count = torch.zeros((num_actions,))

    def get_reward(self,action):
        mean = self.true_values[action]
        std = 1.0
        return normal.Normal(mean,std).sample().item()

    def update_est_value(self,action):
        reward = self.get_reward(action)
        self.action_sample_count[action] += 1.0
        self.est_values[action] = (self.est_values[action] + (1/self.action_sample_count[action])*(reward - self.est_values[action])).item()

    def reset(self):
        self.est_values = torch.zeros((self.num_actions,))
        self.action_sample_count = torch.zeros((self.num_actions,))
    def sampling(self,step):
        raise Exception('Sampling method not defined')

    def run(self,num_steps):
        self.reset()
        cum_reward_avg = torch.zeros((num_steps))
        optimal_action_percent = torch.zeros((num_steps))
        for step in range(1,num_steps+1):
            action = self.sampling(step)
            self.update_est_value(action)
            cum_reward_avg[step-1] = self.reward_average(step)
            optimal_action_percent[step-1] = self.optimal_action(step)
        return cum_reward_avg,optimal_action_percent

    def reward_average(self,step):
        return ((self.est_values*self.action_sample_count).sum()/step).item()
    
    def optimal_action(self,step):
        return (self.action_sample_count[torch.argmax(self.true_values)]/step).item()

class Epsgreedy(Bandit):
    def __init__(self,num_actions,eps):
        super().__init__(num_actions)
        self.eps = eps

    def sampling(self,step):
        sampler = bernoulli.Bernoulli(self.eps).sample().item()
        if sampler == 0:
            return torch.argmax(self.est_values).item()
        else:
            return torch.randint(high=self.num_actions,size=(1,)).item()

class NonStationary(Epsgreedy):
    def __init__(self,num_actions,eps):
        super().__init__(num_actions,eps)

    def sampling(self,step):
        self.true_values = normal.Normal(0.0,(step-1)*0.01).sample([self.num_actions])
        return super().sampling(step)

class ConstantStep(NonStationary):
    def __init__(self,num_actions,eps,step_size):
        super().__init__(num_actions, eps)
        self.step_size = step_size

    def update_est_value(self,action):
        reward = self.get_reward(action)
        self.action_sample_count[action] += 1.0
        self.est_values[action] = (self.est_values[action] + (self.step_size)*(reward - self.est_values[action])).item()
    
class IntialValues(Epsgreedy):
    def __init__(self,num_actions,eps):
        super().__init__(num_actions, eps)

    def reset(self):
        self.est_values = torch.ones((self.num_actions))*5
        self.action_sample_count = torch.zeros((self.num_actions))

class UCBound(Bandit):
    def __init__(self,num_actions,c):
        super().__init__(num_actions)
        self.c = c

    def sampling(self,step):
        return torch.argmax(self.est_values + (self.c*torch.sqrt(math.log(step)/self.action_sample_count))).item()

class Gradient(Bandit):
    def __init__(self,num_actions,alpha,baseline=True):
        super().__init__(num_actions)
        self.alpha = alpha
        self.baseline = baseline
        self._reward_avg = 0.0

    def update_est_value(self,action,step):
        softmax = torch.softmax(self.est_values,dim=0)
        self.action_sample_count[action] += 1.0
        temp = self.est_values[action]
        reward = self.get_reward(action)
        if self.baseline:
            self._reward_avg = (self._reward_avg*(step-1)+ reward)/step
        adj_reward = (reward - self._reward_avg)*self.alpha
        self.est_values = self.est_values - adj_reward*softmax
        self.est_values[action] = temp + adj_reward*(1.0 - softmax[action].item())

    def run(self, num_steps):
        self.reset()
        optimal_action_percent = torch.zeros((num_steps))
        for step in range(1,num_steps+1):
            action = self.sampling(step)
            self.update_est_value(action,step)
            optimal_action_percent[step-1] = self.optimal_action(step)
        return optimal_action_percent
    def sampling(self, step):
        return torch.argmax(self.est_values).item()
