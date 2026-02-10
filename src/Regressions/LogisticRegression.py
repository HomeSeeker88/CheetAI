import numpy as np
import scipy.stats


class LogisticRegression:
    def __init__(self):
        self.coefficients = np.array([])




    def predict(self, x: np.ndarray|list[np.ndarray]):
        # Multiple variables
        if isinstance(x, list):
            X = np.column_stack(x)  # shape: (n_samples, n_features)
            return 1/(1+np.pow(np.e, - (self.coefficients[0] + X @ self.coefficients[1:])))
        # Single variable
        return 1/(1+np.pow(np.e, -(self.coefficients[0] + x * self.coefficients[1])))



    def fit(self, x: np.ndarray | list[np.ndarray], y_values: np.ndarray) -> None:
        ones = np.argwhere(x>0)
        zeros = np.argwhere(x==0)

        prob_ones = np.sum(y_values[ones])/ len(np.extract(x == 1, x))
        prob_zeros = np.sum(y_values[zeros])/len(np.extract(x == 0, x))

        beta_zero = np.log(prob_zeros / (1 - prob_zeros))
        beta_one = np.log(prob_ones / (1 - prob_ones)) - beta_zero
        self.coefficients = np.append(self.coefficients, beta_zero)
        self.coefficients = np.append(self.coefficients, beta_one)










if __name__ == "__main__":
    x = np.array(["smoker","smoker","nonsmoker","nonsmoker","smoker","nonsmoker","nonsmoker","nonsmoker","nonsmoker","smoker","smoker","smoker"])

    y = np.array([0,0,0,0,1,1,1,1,1,0,0,0])
    x_bin = np.where(x == "smoker", 1, 0)
    print(x_bin)
    args =np.argwhere(x_bin > 0)
    log = LogisticRegression()
    print(log.fit(x_bin,y))
    print(log.coefficients)
    print(log.predict(x =np.array([1,1,1,0])))

