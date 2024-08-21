import os

import neat

from neatlab.visualize import Visualize


class Train:
    def __init__(self, experience, backup_filename=None):
        self.experience = experience
        self.backup_filename = backup_filename
        self.config = experience.get_config()
        self.population = self.generate_population()
        self.statistics = self.add_loggers()
        self.visualizer = Visualize(self.experience.get_directory_name())

    def get_backup_filename(self):
        script_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(script_dir)
        complete_filename = os.path.join(
            root_dir,
            'files',
            self.experience.get_directory_name(),
            'backups',
            self.backup_filename
        )
        return complete_filename

    def generate_population(self):
        if self.backup_filename is None:
            return neat.Population(self.config)

        return neat.Checkpointer.restore_checkpoint(self.get_backup_filename())

    def add_loggers(self):
        self.population.add_reporter(neat.StdOutReporter(True))
        statistics = neat.StatisticsReporter()
        self.population.add_reporter(statistics)
        print(self.experience.get_reporter_filename_prefix())
        self.population.add_reporter(neat.Checkpointer(
            generation_interval=25,
            filename_prefix=self.experience.get_reporter_filename_prefix()
        ))
        return statistics

    def evaluation_genomes(self, genomes, config):
        for _, genome in genomes:
            genome.fitness = self.experience(genome, config).get_score()

    def run(self, generation_number=150):
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
