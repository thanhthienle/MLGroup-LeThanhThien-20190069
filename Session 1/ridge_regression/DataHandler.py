import numpy as np


def normalize_and_add_one(X):
    X = np.array(X)
    X_max = np.array([[np.amax(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X_normalized.shape[0])])
    return np.column_stack((ones, X_normalized))


class DataHandler:

    def __init__(self, filename="data.txt"):
        with open(filename) as file:
            self.lines = [line.strip() for line in file.readlines()]
            self.X = [[float(x) for x in line.split()[1:-1]] for line in self.lines]
            self.Y = np.array([float(line.split()[-1]) for line in self.lines])
