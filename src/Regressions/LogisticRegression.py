import numpy as np
import scipy.stats


class LogisticRegression:
    def __init__(self):
        self.coefficients = np.array([])




    def fit(self):
        pass


    def mle(self, x: list[np.ndarray], y_values: np.ndarray) -> None:

        ones = np.count_nonzero(y_values)
        zeros = len(y_values) - ones
        L = scipy.stats.bernoulli(ones/len(y_values))






if __name__ == "__main__":
    y = np.array([0,0,0,0,1,1,1,1,1,0,0,0])
    log = LogisticRegression()
