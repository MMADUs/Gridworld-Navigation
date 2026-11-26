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

    def ucb_action(self, timestep, c=2.0):
        """
        TODO: action = agent.select_action_ucb(timestep=step + 1, c=2.0)
        """
        unvisited_arms = np.where(self.action_counts == 0)[0]

        if len(unvisited_arms) > 0:
            return unvisited_arms[0]

        exploration_bonus = c * np.sqrt(np.log(timestep) / self.action_counts) # ucb scores
        ucb_values = self.action_values + exploration_bonus

        return np.argmax(ucb_values)

    def thompson_action(self):
        sampled_values = np.random.normal(
            loc=self.action_values,
            scale=1.0 / np.sqrt(self.action_counts + 1),
        )

        return np.argmax(sampled_values)

