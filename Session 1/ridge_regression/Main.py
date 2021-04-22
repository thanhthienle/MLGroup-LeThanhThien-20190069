from DataHandler import *
from RidgeRegression import *

data1 = DataHandler()
X, Y = data1.X, data1.Y
del data1

X = normalize_and_add_one(X)
X_train, Y_train = X[:50], Y[:50]
X_test, Y_test = X[50:], Y[50:]

ridge_regression = RidgeRegression()
best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
print('Best LAMBDA:', best_LAMBDA)
W_learned = ridge_regression.fit(
    X_train=X_train, Y_train=Y_train, LAMBDA=best_LAMBDA
)
Y_predicted = ridge_regression.predict(W=W_learned, X_new=X_test)
print(ridge_regression.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted))


