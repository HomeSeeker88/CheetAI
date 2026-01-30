import numpy as np
import random
import typing

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.model import Model


class LinearRegression(Model):

    def __init__(self):
        self.weight = 0
        self.bias = 0

    def fit(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        x_avg = np.mean(x_values)
        y_avg = np.mean(y_values)
        a = np.sum((x_values - x_avg) * (y_values - y_avg)) / np.sum(
            np.power((x_values - x_avg), 2))
        b = y_avg - a * x_avg

        self.weight = a
        self.bias = b

    def predict(self, input: np.ndarray | np.float64) ->  np.ndarray:
        return input * self.weight + self.bias

    def _calculate_mean_square_error(self, y_values: np.ndarray, y_pred_values: np.ndarray) -> np.float64:

        if len(y_values) != len(y_pred_values):
            raise ValueError("Vectors must be the same size!")
        return np.sum(np.power((y_values - y_pred_values),2))/len(y_values)

    def _calculate_mean_error(self, y_values: np.ndarray, y_pred_values: np.ndarray) -> np.float64:
        if len(y_values) != len(y_pred_values):
            raise ValueError("Vectors must be the same size!")
        return np.sum(np.abs(y_values - y_pred_values)) / len(y_values)

    def _calculate_root_mean_squared_error(self, y_values: np.ndarray, y_pred_values: np.ndarray) -> np.float64:
        return np.sqrt(self._calculate_mean_square_error(y_values=y_values,y_pred_values=y_pred_values))

    def _calculate_residual_sum_of_squares(self, y_values: np.ndarray, y_pred_values: np.ndarray) -> np.float64:
        if len(y_values) != len(y_pred_values):
            raise ValueError("Vectors must be the same size!")
        return np.sum(np.pow(y_values - y_pred_values,2))
    def _calculate_total_sum_of_squares(self, y_values: np.ndarray) -> np.float64:
        return np.sum(np.pow(y_values - np.mean(y_values), 2))

    def _calculate_r_square(self, y_values: np.ndarray, y_pred_values: np.ndarray) -> np.float64:
        return 1 - self._calculate_residual_sum_of_squares(y_values=y_values, y_pred_values=y_pred_values) / self._calculate_total_sum_of_squares(y_values=y_values)
    def evaluate(self):
        pass

    def dry_plot(self, x_values: np.ndarray, y_values: np.ndarray, y_pred_values: np.ndarray) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        ax.scatter(x_values, y_values, color=['green'])
        ax.scatter(x_values, y_pred_values, color=['red'])
        ax.plot(x_values, y_pred_values)
        for x, y, y_pred in zip(x_values, y_values, y_pred_values):
            ax.vlines(x=x, ymin = min(y_pred, y), ymax = max(y_pred, y) , color ="red", ls ='--' )
        ax.axvline(0, c='black', ls='--')

        ax.axhline(0, c='black', ls='--')

        return fig, ax



if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-34,-22,-7,0,2.3, 20,-2, 314])

    lr = LinearRegression()
    lr.fit(x_values=x, y_values=y)
    y_pred = lr.predict(x)

    pl = lr.dry_plot(x_values=x,y_values=y,y_pred_values=y_pred)
    #plt.show()

    print(lr._calculate_mean_square_error(y, y_pred))
    print(lr._calculate_mean_error(y, y_pred))
    print(lr._calculate_root_mean_squared_error(y, y_pred))
    print(lr._calculate_residual_sum_of_squares(y, y_pred))
    print(lr._calculate_total_sum_of_squares(y))
    print(lr._calculate_r_square(y, y_pred))
