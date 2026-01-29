import random
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):

    @abstractmethod
    def fit(self, x: np.ndarray, y:np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, input: np.ndarray | np.float64) ->  np.ndarray | np.float64:
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def split_train_test(self, input: np.ndarray, proportion: float) -> tuple([np.ndarray, np.ndarray]):
        shuffled_vec = random.shuffle(input)
        split_index= np.floor(proportion * len(input))
        train_set = shuffled_vec[:split_index]
        test_set = shuffled_vec[split_index:]
        return train_set, test_set