import torch
import torch.nn as nn
from torch.distributions import categorical


class Grid(object):
    """
    This is GridWorld env.\n
    Attributes:\n
        height, width : dims of Grid Env.\n
        state_prob : Here state transition prob is 1.0 as each action on state leads to single specific state all the time.\n
        __pointer : Agent's current state.\n
        state_values : State values\n
        action_mapper : Dict of action name and value to move pointer.\n
        gamma : Discount factor\n
        theta : Threshold for policy evaluation
    """

    def __init__(self, width, height):

        assert width == height, "Grid width and height must be equal"
        self.height = height
        self.width = width

        self.action_mapper = {'north': torch.tensor([1, 0], dtype=torch.int),
                              'south': torch.tensor([-1, 0], dtype=torch.int),
                              'east': torch.tensor([0, 1], dtype=torch.int),
                              'west': torch.tensor([0, -1], dtype=torch.int)}
        self.__pointer = torch.tensor([0, 0], dtype=torch.int)
        self.state_values = torch.zeros(self.height * self.width)
        self.state_prob = 1.0
        self.gamma = 1.0
        self.theta = 0.000001
        # Using random policy, all actions are equal probable.
        self.policy = torch.ones(self.state_values.numel(), 4)/4

    @property
    def pointer(self):
        return self.__pointer[0].item(), self.__pointer[1].item()

    def next_state_reward(self, state, action):
        """Returns next state and reward obtained"""
        # State to grid position mapping.
        self.__pointer = torch.tensor(
            [int(state % self.width), int(state/self.width)])
        # Terminal states
        if self.pointer == (0, 0) or self.pointer == (self.width-1, self.height-1):
            self.state_values[state] = 0.0
            self.__reward = 0.0

        else:
            temp = self.__pointer + self.action_mapper[action]
            self.__reward = -1.0
            # Restricting agent movement to inside of Grid
            if -1 not in temp and temp[0] < self.height and temp[1] < self.width:
                self.__pointer = temp
        # Grid position to state mapping
        state = self.pointer[0] * self.width + self.pointer[1]
        return self.state_values[state], self.__reward

    def policy_eval(self):
        """ Iterative Policy Evaluation with in place state values. (Ref:Topic 4.1 in Sutton and Burto) """
        steps = 0
        while True:
            diff = torch.tensor(0.0)
            for state in range(self.state_values.numel()):
                v = torch.tensor(0.0)
                for k, action in enumerate(self.action_mapper):
                    next_state, reward = self.next_state_reward(state, action)
                    v += self.policy[state, k] * self.state_prob * (
                        reward + self.gamma * next_state)
                diff = torch.max(diff, torch.absolute(
                    self.state_values[state]-v))
                self.state_values[state] = v.clone()
            steps += 1
            if diff < self.theta:
                print("Total Steps:", steps)
                break
        return self.state_values.reshape(self.width, self.height)

    def policy_eval_long(self):
        """ Iterative Policy Evaluation with two arrays. (Ref:Topic 4.1 in Sutton and Burto) """
        steps = 0
        while True:
            diff = torch.tensor(0.0)
            temp = torch.zeros(self.state_values.shape)
            for state in range(self.state_values.numel()):
                for k, action in enumerate(self.action_mapper):
                    self.__pointer = torch.tensor([state])
                    next_state, reward = self.next_state_reward(state, action)
                    temp[state] += self.policy[state, k] * self.state_prob * (
                        reward + self.gamma * next_state)

                diff = torch.max(diff, torch.absolute(
                    self.state_values[state]-temp[state]))
            self.state_values = temp.clone()
            steps = steps + 1
            if diff < self.theta:
                print("Total Steps:", steps)
                break
        return self.state_values.reshape(self.width, self.height)


# S = Grid(5, 5)
# G = Grid(5, 5)
# print(S.policy_eval())
# print(G.policy_eval_long())
