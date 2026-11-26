import numpy as np


def calculate_return(rewards, gamma=0.9):
    total_return = 0.0

    for t, reward in enumerate(rewards):
        total_return += (gamma ** t) * reward

    return total_return


if __name__ == "__main__":
    rewards = [-1, -1, -1, -1, 10]

    undiscounted = calculate_return(rewards, gamma=1.0)
    discounted = calculate_return(rewards, gamma=0.9)

    print("Rewards:", rewards)
    print("Undiscounted return:", undiscounted)
    print("Discounted return:", discounted)