import numpy as np
from gymnasium.envs.classic_control import AcrobotEnv

class CustomAcrobotEnv(AcrobotEnv):
    def __init__(self, goal_height=0.0, **kwargs):
        super(AcrobotEnv, self).__init__(**kwargs)
        self.goal_height = goal_height
        self.max_height = -np.inf

    def current_height(self):
        length_1 = self.LINK_LENGTH_1
        length_2 = self.LINK_LENGTH_2

        return -(length_1 * np.cos(self.state[0]) + length_2 * np.cos(self.state[0] + self.state[1]))

    def step(self, action):
        observation, reward, done, truncated, info = super(CustomAcrobotEnv, self).step(action)

        # Update max height reached
        if self.current_height() > self.max_height:
            self.max_height = self.current_height()

        # Custom reward based on max height reached
        reward += self.max_height

        # Check if the custom goal height has been reached
        done = self.current_height() >= self.goal_height

        return observation, reward, done, truncated, info

    def reset(self, *args, **kwargs):
        self.max_height = -np.inf
        return super(CustomAcrobotEnv, self).reset(*args, **kwargs)
