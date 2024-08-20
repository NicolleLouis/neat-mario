import neat

from neatlab.visualize import Visualize


class Train:
    def __init__(self, experience):
        self.experience = experience
        self.config = experience.get_config()
        self.population = neat.Population(self.config)
        self.statistics = self.add_loggers()
        self.visualizer = Visualize(self.experience.get_directory_name())

    def add_loggers(self):
        self.population.add_reporter(neat.StdOutReporter(True))
        statistics = neat.StatisticsReporter()
        self.population.add_reporter(statistics)
        return statistics
    
    def evaluation_genomes(self, genomes, config):
        for _, genome in genomes:
            genome.fitness = self.experience(genome, config).get_score()
    
    def run(self, generation_number = 200):
        winner = self.population.run(self.evaluation_genomes, generation_number)
        self.output_winner(winner)
        self.output_graphs(winner)

    @staticmethod
    def output_winner(winner):
        print('\nBest genome:\n{!s}'.format(winner))

    def output_graphs(self, winner):
        node_names = self.experience.get_node_names()
        self.visualizer.draw_net(self.config, winner, True, node_names=node_names)
        self.visualizer.plot_stats(self.statistics, ylog=False, view=True)
        self.visualizer.plot_species(self.statistics, view=True)