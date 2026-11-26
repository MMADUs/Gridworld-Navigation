import numpy as np


class RandomAgent:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def random_action(self):
        return np.random.randint(self.n_actions)