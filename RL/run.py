#%%
import torch
from torch.distributions import normal,bernoulli
import matplotlib.pyplot as plt
import model
import importlib
importlib.reload(model)
from model import Epsgreedy, NonStationary, ConstantStep, IntialValues, UCBound, Gradient
torch.manual_seed(14)

# %%
def plot_results(model,steps,num_actions,eps,runs):
    total_reward = {}
    total_optimal = {}
    _,(ax1, ax2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.5)
    for e in eps:
        for k in range(runs):
            epsbandit = model(num_actions=num_actions,eps=e)

            if e not in total_optimal.keys():
                total_reward[e],total_optimal[e]= epsbandit.run(steps)
            else:
                a,b = epsbandit.run(steps)
                total_reward[e] += a
                total_optimal[e] += b
        total_optimal[e] = total_optimal[e]/runs
        total_reward[e] = total_reward[e]/runs
        ax1.plot(total_reward[e],label = 'eps ='+str(e))
        ax2.plot(total_optimal[e],label = 'eps ='+str(e))

    ax1.set_xlabel('Steps',color= 'yellow')
    ax1.set_ylabel('Reward',color = 'yellow')
    ax1.legend(loc = 'best')
    ax2.set_xlabel('Steps',color= 'yellow')
    ax2.set_ylabel('Optimal Action',color = 'yellow')
    ax2.legend(loc = 'best')
    plt.show()

# %%
steps = 1000
eps = [0.0]
num_actions = 20
runs = 5
#%%
#plot_results(Epsgreedy,steps,num_actions,eps,runs)
#plot_results(NonStationary,steps,num_actions,eps,runs)
plot_results(IntialValues,steps,num_actions,eps,runs)
# %%
def plot_results_const_step(model,steps,num_actions,eps,runs,step_size):
    total_reward = {}
    total_optimal = {}
    _,(ax1, ax2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.5)
    for i in step_size:
        for k in range(runs):
            epsbandit = model(num_actions=num_actions,eps=eps,step_size = i)

            if i not in total_optimal.keys():
                total_reward[i],total_optimal[i]= epsbandit.run(steps)
            else:
                a,b = epsbandit.run(steps)
                total_reward[i] += a
                total_optimal[i] += b
        total_optimal[i] = total_optimal[i]/runs
        total_reward[i] = total_reward[i]/runs
        ax1.plot(total_reward[i],label = 'step = '+str(i))
        ax2.plot(total_optimal[i],label = 'eps = '+str(i))

    ax1.set_xlabel('Steps',color= 'yellow')
    ax1.set_ylabel('Reward',color = 'yellow')
    ax1.legend(loc = 'best')
    ax2.set_xlabel('Steps',color= 'yellow')
    ax2.set_ylabel('Optimal Action',color = 'yellow')
    ax2.legend(loc = 'best')
    plt.show()


# %%
eps = 0.1
steps = 1000
num_actions = 20
runs = 5
step_size = [0.1,0.15,0.2]
plot_results_const_step(ConstantStep,steps,num_actions,eps,runs,step_size)

# %%

def plot_results_UCB(model,steps,num_actions,runs,c):
    total_reward = {}
    total_optimal = {}
    _,(ax1, ax2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.5)
    for i in c:
        for k in range(runs):
            epsbandit = model(num_actions=num_actions,c=i)

            if i not in total_optimal.keys():
                total_reward[i],total_optimal[i]= epsbandit.run(steps)
            else:
                a,b = epsbandit.run(steps)
                total_reward[i] += a
                total_optimal[i] += b
        total_optimal[i] = total_optimal[i]/runs
        total_reward[i] = total_reward[i]/runs
        ax1.plot(total_reward[i],label = 'step = '+str(i))
        ax2.plot(total_optimal[i],label = 'eps = '+str(i))

    ax1.set_xlabel('Steps',color= 'yellow')
    ax1.set_ylabel('Reward',color = 'yellow')
    ax1.legend(loc = 'best')
    ax2.set_xlabel('Steps',color= 'yellow')
    ax2.set_ylabel('Optimal Action',color = 'yellow')
    ax2.legend(loc = 'best')
    plt.show()

# %%
steps = 500
num_actions = 20
runs = 2
c = [0.8,2]
plot_results_UCB(UCBound,steps,num_actions,runs,c)

#%%
def plot_results_Gradient(model,steps,num_actions,runs,alpha):
    total_optimal = {}
    for i in alpha:
        for k in range(runs):
            epsbandit = model(num_actions=num_actions,alpha=i,baseline=True)
            if i not in total_optimal.keys():
                total_optimal[i]= epsbandit.run(steps)
            else:
                b = epsbandit.run(steps)
                total_optimal[i] += b
        total_optimal[i] = total_optimal[i]/runs
        plt.plot(total_optimal[i],label = 'alpha = '+str(i))

    plt.xlabel('Steps',color= 'yellow')
    plt.ylabel('Optimal Action',color = 'yellow')
    plt.legend(loc = 'best')
    plt.show()

# %%
steps = 1000
num_actions = 10
runs = 5
alpha = [0.1,0.4]
plot_results_Gradient(Gradient,steps,num_actions,runs,alpha)
# %%
