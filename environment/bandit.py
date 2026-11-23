import numpy as np


class BanditEnvironment:
    def __init__(self, n_arms=4):
        self.n_arms = n_arms
        self.reward_means = np.random.randn(n_arms)
        self.reward_std = 1.0

    def step(self, action):
        reward = np.random.normal(
            loc=self.reward_means[action],
            scale=self.reward_std,
        )

        return reward