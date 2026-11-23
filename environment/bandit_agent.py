import numpy as np

from environment.bandit import BanditEnvironment


class BanditAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.action_counts = np.zeros(n_arms)  # n times each arm has been selected
        self.action_values = np.zeros(n_arms)  # estimation value of each arm

    def update(self, action, reward):
        self.action_counts[action] += 1

        n = self.action_counts[action]
        old_value = self.action_values[action]

        new_value = old_value + (reward - old_value) / n # sample-average estimation
        self.action_values[action] = new_value

    def greedy_action(self):
        action = np.argmax(self.action_values)
        return action

    def epsilon_greedy_action(self, epsilon=0.1):
        random = np.random.rand() # uniform 0-1

        if random < epsilon:
            action = np.random.randint(self.n_arms)
        else:
            action = np.argmax(self.action_values)

        return action
