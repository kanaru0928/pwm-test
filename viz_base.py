import abc
import numpy as np


class VizBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def render(self, data: np.ndarray, sample_rate: int):
        """Render the visualization with the provided data."""
        pass

    @abc.abstractmethod
    def save(self, data: np.ndarray, sample_rate: int, filepath: str):
        """Save the visualization to the specified file path."""
        pass
