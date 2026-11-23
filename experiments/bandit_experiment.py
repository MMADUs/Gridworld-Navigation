import numpy as np
import matplotlib.pyplot as plt

from environment.bandit import BanditEnvironment
from environment.bandit_agent import BanditAgent


def experiment_1():
    """
    gaussian multi-armed bandit environment
    """
    env = BanditEnvironment(n_arms=4)

    print("true reward means:", env.reward_means)

    for arm in range(env.n_arms):
        rewards = []

        for _ in range(5):
            reward = env.step(arm)
            rewards.append(reward)

        print(f"arm {arm}: {rewards}")
        print(f"avg reward: {sum(rewards) / len(rewards):.3f}")


def experiment_2():
    """
    sample-average action-value estimation
    """
    env = BanditEnvironment(n_arms=4)
    agent = BanditAgent(n_arms=4)

    print("true reward means:", env.reward_means)

    for arm in range(env.n_arms):
        for _ in range(1000):
            reward = env.step(arm)
            agent.update(action=arm, reward=reward)

    print("agent's estimated values:", agent.action_values)
    print("action selection counts:", agent.action_counts)


def experiment_3():
    """
    greedy action selection (exploitation)
    """
    env = BanditEnvironment(n_arms=4)
    agent = BanditAgent(n_arms=4)

    n_steps = 1000

    for _ in range(n_steps):
        action = agent.greedy_action()
        reward = env.step(action)
        agent.update(action=action, reward=reward)

    print("true reward means:", env.reward_means)
    print("estimated values:", agent.action_values)
    print("action counts:", agent.action_counts)


def experiment_4():
    """
    epsilon greedy action selection (exploration)
    """
    env = BanditEnvironment(n_arms=4)
    agent = BanditAgent(n_arms=4)

    n_steps = 1000

    for _ in range(n_steps):
        action = agent.epsilon_greedy_action(epsilon=0.1)
        reward = env.step(action)
        agent.update(action=action, reward=reward)

    print("true reward means:", env.reward_means)
    print("estimated values:", agent.action_values)
    print("action counts:", agent.action_counts)


def run_bandit(epsilon=0.1, n_steps=1000):
    """
    bandit performance measurement
    """
    env = BanditEnvironment(n_arms=4)
    agent = BanditAgent(n_arms=4)

    reward_history = np.zeros(n_steps)

    for step in range(n_steps):
        action = agent.epsilon_greedy_action(epsilon=epsilon)
        reward = env.step(action)
        agent.update(action=action, reward=reward)

        reward_history[step] = reward

    # calculate performance metrics
    cumulative_rewards = np.cumsum(reward_history)
    average_rewards = cumulative_rewards / np.arange(1, n_steps + 1)

    print("true reward means:", env.reward_means)
    print("estimated values:", agent.action_values)
    print("action counts:", agent.action_counts)

    print("total reward:", cumulative_rewards[-1])
    print("average reward:", average_rewards[-1])

    optimal_reward = np.max(env.reward_means)

    plt.figure(figsize=(10, 5))
    plt.plot(average_rewards, label="agent average reward")
    plt.axhline(y=optimal_reward, linestyle="--", label="optimal expected reward")
    plt.xlabel("steps")
    plt.ylabel("average reward")
    plt.title(f"Epsilon-Greedy (epsilon={epsilon})")
    plt.legend()
    plt.grid()
    plt.show()


def regret_tracking_bandit(epsilon=0.1, n_steps=1000):
    """
    bandit performance measurement
    """
    env = BanditEnvironment(n_arms=4)
    agent = BanditAgent(n_arms=4)

    reward_history = np.zeros(n_steps)
    regret_history = np.zeros(n_steps)

    optimal_reward = np.max(env.reward_means)

    for step in range(n_steps):
        action = agent.epsilon_greedy_action(epsilon=epsilon)
        reward = env.step(action)
        agent.update(action=action, reward=reward)

        reward_history[step] = reward
        regret_history[step] = optimal_reward - env.reward_means[action]

    # calculate performance metrics
    cumulative_rewards = np.cumsum(reward_history)
    cumulative_regret = np.cumsum(regret_history)

    average_rewards = cumulative_rewards / np.arange(1, n_steps + 1)

    print("true reward means:", env.reward_means)
    print("estimated values:", agent.action_values)
    print("action counts:", agent.action_counts)

    print("cumulative regret:", cumulative_regret[-1])
    print("total reward:", cumulative_rewards[-1])
    print("average reward:", average_rewards[-1])

    plt.figure(figsize=(10, 5))
    plt.plot(average_rewards, label="agent average reward")
    plt.axhline(y=optimal_reward, linestyle="--", label="optimal expected reward")
    plt.xlabel("steps")
    plt.ylabel("average reward")
    plt.title(f"Epsilon-Greedy (epsilon={epsilon})")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_regret, label="cumulative regret")
    plt.xlabel("steps")
    plt.ylabel("cumulative regret")
    plt.title(f"Cumulative Regret (epsilon={epsilon})")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # experiment_1()
    # print("\n")
    # experiment_2()
    # print("\n")
    # experiment_3()
    # print("\n")
    # experiment_4()
    # print("\n")
    # run_bandit(epsilon=0.3, n_steps=5000)
    # print("\n")
    regret_tracking_bandit(epsilon=0.01, n_steps=5000)
