import numpy as np
from abc import ABC, abstractmethod

class BaseGame(ABC):
    """
    Base Interface for Environment/Game.
    """

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def get_rewards(self):
        pass

    @staticmethod
    @abstractmethod
    def get_game_list():
        pass
