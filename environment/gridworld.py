import numpy as np 


class GridWorld:
    def __init__(self, size=5, max_steps=100):
        self.size = size
        self.max_steps = max_steps

        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)

        self.actions = {
            0: (-1, 0), # up
            1: (1, 0),  # down
            2: (0, -1), # left
            3: (0, 1),  # right
        }

        self.reset()

    def reset(self):
        self.state = self.start_pos
        self.steps = 0

        return self.state 

    def move_to_next_state(self, state, action):
        row, col = state 
        row_change, col_change = self.actions[action]

        new_row = np.clip(row + row_change, 0, self.size - 1)
        new_col = np.clip(col + col_change, 0, self.size - 1)

        return int(new_row), int(new_col)

    def get_reward(self, next_state):
        return 10 if next_state == self.goal_pos else -1

    def do_transition(self, state, action):
        if state == self.goal_pos:
            return state, 0, True

        next_state = self.move_to_next_state(state, action)
        reward = self.get_reward(next_state)

        is_done = next_state == self.goal_pos

        return next_state, reward, is_done

    def step(self, action):
        next_state, reward, is_done = self.do_transition(self.state, action)

        self.state = next_state
        self.steps += 1

        reached_limit = self.steps >= self.max_steps and not is_done
        is_finish = is_done or reached_limit

        return self.state, reward, is_finish

    def render(self):
        grid = np.full((self.size, self.size), ".", dtype=str)

        grid[self.goal_pos] = "G"
        grid[self.state] = "A"

        for row in grid:
            print(" ".join(row))

        print()