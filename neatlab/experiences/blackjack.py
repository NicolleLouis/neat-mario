import gymnasium

from neatlab.experiences.experience import Experience


class Blackjack(Experience):
    CONFIG_NAME = 'config-blackjack'
    DIRECTORY_NAME = 'blackjack'
    NODE_NAMES = {-1: 'OwnScore', -2: 'Dealer score', -3: 'Usable Ace', 0: 'Stop', 1: 'Hit'}

    def get_score(self, game_limit=100) -> int:
        score = 0
        has_lost = False

        while not has_lost or score >= game_limit:
            is_win = self.run_blackjack_game()
            if is_win:
                score += 1
            else:
                has_lost = True

        return score

    def generate_environment(self):
        return gymnasium.make("Blackjack-v1", sab=True)

    # Run a blackjack game, if the game is won or draw, return True, else False
    def run_blackjack_game(self) -> bool:
        observation, info = self.environment.reset()
        done = False
        result = None

        while not done:
            action = self.compute_action(observation)
            next_observation, reward, terminated, truncated, info = self.environment.step(action)

            done = terminated or truncated
            result = reward

        if result in [1, 0]:
            return True
        if result in [-1]:
            return False

        print(result)
        raise ValueError("Incorrect result")

    def compute_action(self, observation):
        output = self.net.activate(observation)
        action = output.index(max(output))
        return action
