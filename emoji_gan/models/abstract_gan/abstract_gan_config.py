import abc
import json


class AbstractGANConfig(abc.ABC):
    def __init__(self):
        pass

    def save(self, config_filepath):
        with open(config_filepath, 'w') as f:
            json.dump(self.__dict__, f, sort_keys=True, indent=True)

    @abc.abstractmethod
    def restore(self, config_filepath):
        pass
