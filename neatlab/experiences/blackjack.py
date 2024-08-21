import random

import gymnasium

from neatlab.experiences.experience import Experience


class Blackjack(Experience):
    CONFIG_NAME = 'config-blackjack'
    DIRECTORY_NAME = 'blackjack'
    NODE_NAMES = {-1: 'OwnScore', -2: 'Dealer score', -3: 'Usable Ace', 0: 'Stop', 1: 'Hit'}

    def get_score(self, game_number=100) -> int:
        score = 0

        for _ in range(game_number):
            score += self.run_blackjack_game()

        return score

    def run_example(self):
        for _ in range(10):
            result = self.run_blackjack_game()
            print(f"result: {self.translate_result(result)}")

    @staticmethod
    def translate_result(result):
        if result == 1:
            return "Victory"
        elif result == 0:
            return "Draw"
        elif result == -1:
            return "Defeat"

        raise ValueError(f"Unknown result: {result}")

    def generate_environment(self, human=False):
        params = {
            'id': "Blackjack-v1",
            'sab': True,
        }
        if human:
            params['render_mode'] = "human"

        return gymnasium.make(**params)

    def run_blackjack_game(self) -> bool:
        observation, info = self.environment.reset()
        done = False
        result = None

        while not done:
            action = self.compute_action(observation)
            next_observation, reward, terminated, truncated, info = self.environment.step(action)

            done = terminated or truncated
            result = reward

        return result

    @staticmethod
    def preprocess_observation(observation):
        return observation[0] - 11, observation[1] - 11, observation[2]

    def compute_action(self, observation):
        cleaned_observation = self.preprocess_observation(observation)
        output = self.net.activate(cleaned_observation)
        if output == [0, 0]:
            return random.randint(0, 1)
        action = output.index(max(output))
        return action
