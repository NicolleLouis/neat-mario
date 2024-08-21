import gymnasium

from neatlab.experiences.experience import Experience


class Acrobot(Experience):
    CONFIG_NAME = 'config-acrobot'
    DIRECTORY_NAME = 'acrobot'
    NODE_NAMES = {
        -1: 'Cosine 1',
        -2: 'Sine 1',
        -3: 'Cosine 2',
        -4: 'Sine 2',
        -5: 'Velocity 1',
        -6: 'Velocity 2',
        0: 'Left',
        1: 'Stop',
        2: 'Right',
    }

    def run_example(self):
        self.run()

    def get_score(self):
        return self.run()

    def generate_environment(self, human=False):
        params = {
            'id': "Acrobot-v1",
        }
        if human:
            params['render_mode'] = "human"
        return gymnasium.make(**params)

    def run(self):
        observation, info = self.environment.reset()
        done = False
        time_taken = 0
        reward = 0

        while not done:
            time_taken += 1
            action = self.compute_action(observation)
            next_observation, reward, terminated, truncated, info = self.environment.step(action)

            if terminated:
                reward += 100

            done = terminated or truncated
            observation = next_observation

        return reward - 0.5 * time_taken

    def compute_action(self, observation):
        output = self.net.activate(observation)
        action = output.index(max(output))
        return action

    @staticmethod
    def compute_height(observation):
        cosine_1 = observation[0]
        sine_1 = observation[1]
        cosine_2 = observation[2]
        sine_2 = observation[3]

        return - cosine_1 - (cosine_1 * cosine_2 - sine_1 * sine_2)
