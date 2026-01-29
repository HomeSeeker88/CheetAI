import numpy as np
import random
import typing

from matplotlib import pyplot as plt

from src.model import Model


class LinearRegression(Model):

    def __init__(self):
        self.weight = 0
        self.bias = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_avg = np.mean(x)
        y_avg = np.mean(y)
        a = np.sum([(xi - x_avg) * (yi - y_avg) for xi, yi in zip(x, y)]) / np.sum(
            [np.power((xi - x_avg), 2) for xi in x])
        b = y_avg - a * x_avg

        self.weight = a
        self.bias = b

    def predict(self, input: np.ndarray | np.float64) ->  np.ndarray:
        return input * self.weight + self.bias

    def evaluate(self):
        pass



if __name__ == "__main__":
    x = np.array([3,4,5,6])
    y = np.array([-10,24,37,43])

    lr = LinearRegression()
    lr.fit(x=x, y=y)
    y_pred = lr.predict(x)
    plt.plot(x,y_pred)
    plt.show()