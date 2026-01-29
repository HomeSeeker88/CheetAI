import numpy as np
import random
from src.model import Model


class LinearRegression(Model):

    def __init__(self):
        self.weight = 0
        self.bias = 0


    def fit(self, input: np.ndarray):
