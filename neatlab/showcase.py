import pickle


class Showcase:
    def __init__(self, experience):
        self.experience = experience
        self.config = experience.get_config()
        self.winner = self.get_winner()
        self.run()

    def get_winner(self, winner_filename='winner.pkl'):
        winner_file = self.experience.get_winner_filename(winner_filename)
        with open(winner_file, 'rb') as f:
            winner = pickle.load(f)
            return winner

    def run(self):
        self.experience(self.winner, self.config).showcase()
