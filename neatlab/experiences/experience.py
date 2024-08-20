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

    @abstractmethod
    def generate_environment(self):
        pass

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
