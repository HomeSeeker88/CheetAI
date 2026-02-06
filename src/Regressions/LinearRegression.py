import numpy as np
import random
import typing

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import shapiro

from src.Summaries.Summaries import LinearRegressionSummary
from src.model import Model


class LinearRegression(Model):

    def __init__(self):
        self.weight = 0
        self.bias = 0
        self.coefficients = np.array([])

    def fit(self, x_values: np.ndarray | list[np.ndarray], y_values: np.ndarray) -> None:
        self.y_avg = np.mean(y_values)
        self.bias = self.y_avg
        if isinstance(x_values, list):
            x_values = np.column_stack(x_values)
            ones = np.ones((x_values.shape[0], 1))
            x_values = np.hstack((ones, x_values))
            x_values_t = np.transpose(x_values)
            self.coefficients = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_values_t, x_values)), x_values_t), y_values)
        else:
            x_avg = np.mean(x_values)
            a = self._calculate_coefficient(x_values=x_values, y_values=y_values)
            b = self.y_avg - a * x_avg
            self.coefficients = np.array([b, a])
        self.bias = self.coefficients[0]
        self.train_x = x_values
        self.train_y = y_values

    def _calculate_coefficient(self, x_values: np.ndarray, y_values: np.ndarray) -> np.float64:
        x_avg = np.mean(x_values)
        y_avg = np.mean(y_values)
        a = np.sum((x_values - x_avg) * (y_values - y_avg)) / np.sum(
            np.power((x_values - x_avg), 2))
        return a

    def predict(self, input: np.ndarray | list[np.ndarray]) -> np.ndarray:
        # Multiple variables
        if isinstance(input, list):
            X = np.column_stack(input)  # shape: (n_samples, n_features)
            return self.coefficients[0] + X @ self.coefficients[1:]

        # Single variable
        return self.coefficients[0] + input * self.coefficients[1]

    def _calculate_mean_square_error(self,  y_pred_values: np.ndarray) -> np.float64:

        if len(self.train_y) != len(y_pred_values):
            raise ValueError("Vectors must be the same size!")
        return np.sum(np.power((self.train_y - y_pred_values),2))/len(self.train_y)

    def _calculate_mean_absolute_error(self, y_pred_values: np.ndarray) -> np.float64:
        if len(self.train_y) != len(y_pred_values):
            raise ValueError("Vectors must be the same size!")
        return np.sum(np.abs(self.train_y - y_pred_values)) / len(self.train_y)

    def _calculate_root_mean_squared_error(self,  y_pred_values: np.ndarray) -> np.float64:
        return np.sqrt(self._calculate_mean_square_error(y_pred_values=y_pred_values))

    def _calculate_residual_sum_of_squares(self, y_pred_values: np.ndarray) -> np.float64:
        if len(self.train_y) != len(y_pred_values):
            raise ValueError("Vectors must be the same size!")
        return np.sum(np.pow(self.train_y - y_pred_values,2))
    def _calculate_total_sum_of_squares(self) -> np.float64:
        return np.sum(np.pow(self.train_y - np.mean(self.train_y), 2))

    def _calculate_r_square(self, y_pred_values: np.ndarray) -> np.float64:
        return 1 - self._calculate_residual_sum_of_squares(y_pred_values=y_pred_values) / self._calculate_total_sum_of_squares()

    def _check_error_normality(self, y_pred_values: np.ndarray):
        return True if shapiro(np.abs(self.train_y - y_pred_values))[1] > 0.05 else False
    def evaluate(self, y_pred_values: np.ndarray):
        self.summary = LinearRegressionSummary(rmse = self._calculate_total_sum_of_squares(),
                                               mse = self._calculate_mean_square_error(y_pred_values=y_pred_values),
                                               mae = self._calculate_mean_absolute_error(y_pred_values=y_pred_values),
                                               r_squared = self._calculate_r_square(y_pred_values=y_pred_values),
                                               error_normality = self._check_error_normality(y_pred_values=y_pred_values)
                                               )

    def dry_plot(self, x_values: np.ndarray, y_values: np.ndarray, y_pred_values: np.ndarray, draw_axis: bool = False) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        ax.scatter(x_values, y_values, color=['green'])
        ax.scatter(x_values, y_pred_values, color=['red'])
        ax.plot(x_values, y_pred_values)
        for x, y, y_pred in zip(x_values, y_values, y_pred_values):
            ax.vlines(x=x, ymin = min(y_pred, y), ymax = max(y_pred, y) , color ="red", ls ='--' )

        if draw_axis:
            ax.axvline(0, c='black', ls='--')
            ax.axhline(0, c='black', ls='--')

        return fig, ax



if __name__ == "__main__":
    x = np.array([168, 188, 165, 195, 165, 182, 173, 190])
    y = np.array([70, 82, 90, 106, 47, 85, 71, 94])

    lr = LinearRegression()
    lr.fit(x_values=x, y_values=y)
    y_pred = lr.predict(x)

    pl = lr.dry_plot(x_values=x,y_values=y,y_pred_values=y_pred)
    print(lr.coefficients)
    print(lr.bias)
    print(y_pred)
    #plt.show()

    lr.evaluate(y_pred_values=y_pred)
    print(lr.summary)

    x1 = np.array([1,2,3])
    x2 = np.array([1,2,3])
    print(np.multiply(x1,x2))

    x1 = np.array([168, 188, 165, 195, 165, 182, 173, 190])
    x2 = np.array([70, 85, 88, 100, 45, 84, 72, 100])
    y = np.array([70, 82, 90, 106, 47, 85, 71, 94])

    lr_multiple_vars = LinearRegression()
    lr_multiple_vars.fit(x_values = [x1, x2], y_values=y)
    y_pred_multiple_vars = lr_multiple_vars.predict([x1, x2])
    print(lr_multiple_vars.coefficients, lr_multiple_vars.bias)
    print(y_pred_multiple_vars)
    lr_multiple_vars.evaluate(y_pred_values=y_pred_multiple_vars)
    print(lr_multiple_vars.summary)