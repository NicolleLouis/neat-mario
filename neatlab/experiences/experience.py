import os
from abc import ABC, abstractmethod

import neat


class Experience(ABC):
    CONFIG_NAME = None
    DIRECTORY_NAME = None
    NODE_NAMES = None

    def __init__(self, genomes, config):
        self.net = neat.nn.FeedForwardNetwork.create(genomes, config)
        self.environment = self.generate_environment()

    @abstractmethod
    def get_score(self):
        pass

    @staticmethod
    @abstractmethod
    def generate_environment(human=False):
        pass

    def showcase(self):
        self.environment = self.generate_environment(human=True)
        self.run_example()

    @abstractmethod
    def run_example(self):
        pass

    def generate_net(self, genomes, config):
        self.net = neat.nn.FeedForwardNetwork.create(genomes, config)

    @classmethod
    def get_config_path(cls):
        if cls.CONFIG_NAME is None:
            raise ValueError("CONFIG_NAME is not set")

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, cls.CONFIG_NAME)
        return config_path

    @classmethod
    def get_config(cls):
        config_path = cls.get_config_path()
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

    @classmethod
    def get_node_names(cls):
        if cls.NODE_NAMES is None:
            raise NotImplementedError("Should have a node names computation implemented")
        return cls.NODE_NAMES

    @classmethod
    def get_directory_name(cls):
        if cls.DIRECTORY_NAME is None:
            raise NotImplementedError("Should have a directory name computation implemented")
        return cls.DIRECTORY_NAME

    @classmethod
    def get_reporter_filename_prefix(cls):
        root_dir = cls.get_root_dir()
        complete_filename = os.path.join(
            root_dir,
            'files',
            cls.get_directory_name(),
             'backups/neat-checkpoint-'
        )
        return complete_filename

    @classmethod
    def get_root_dir(cls):
        script_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(script_dir)
        root_dir = os.path.dirname(root_dir)
        return root_dir

    @classmethod
    def get_winner_filename(cls, winner_filename='winner.pkl'):
        root_dir = cls.get_root_dir()
        return os.path.join(
            root_dir,
            'files',
            cls.get_directory_name(),
            winner_filename
        )

    @classmethod
    def get_backup_directory(cls):
        root_dir = cls.get_root_dir()
        return os.path.join(
            root_dir,
            'files',
            cls.get_directory_name(),
            'backups'
        )
