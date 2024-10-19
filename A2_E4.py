import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

#Loads the data from red wine
data = pd.read_csv('wine+quality/winequality-red.csv', sep = ';')

#splits the data to train set and test set
train, test = train_test_split(data)

#drops the quality column of train set
phi_train = train.drop('quality', axis = 1).values

# the target values (quality) of test set
actual_train = train['quality'].values

#computes the closed form solution of sum-of-squares
w_star_train = np.linalg.inv(phi_train.T @ phi_train) @ phi_train.T @ actual_train

#computes the prediction from learned weight parameter w
predict_train = phi_train @ w_star_train

#Plots the actual and prediction
plt.scatter(actual_train, predict_train)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Quality ")
plt.show()

#Calculates root-mean-square on the train
rms_train = np.sqrt(mean_squared_error(actual_train, predict_train))

#drops the quality column of test set
phi_test = test.drop('quality', axis = 1).values

# the target values (quality) of test set
actual_test = test['quality'].values

#computes the closed form solution of sum-of-squares of test set
w_star_test = np.linalg.inv(phi_test.T @ phi_test) @ phi_test.T @ actual_test

#computes the prediction from learned weight parameter w on test set
predict_test = phi_test @ w_star_test

#Calculates root-mean-square on the test
rms_test = np.sqrt(mean_squared_error(actual_test, predict_test))

print(f"Root-Mean-Squared-Error on Train Set: {rms_train:.3f}")
print(f"Root-Mean-Squared-Error on Test Set: {rms_test:.3f}")

#LMS algorithm

#finds the number of features
n_phi_train = len(phi_train[0])

#finds the number of samples of trainset
n_train = len(phi_train)

#initialize w_0
w_0 = np.random.randn(n_phi_train)

#intiailize the step size
step_size = 0.00001

w = w_0

#LMS algorithm with 100 iterations
for i in range(100):
    #computes the prediction using w_0 on training set
    predict_w_train = phi_train @ w

    #computes the error
    error_train = actual_train - predict_w_train

    #computes the gradient
    gradient = (phi_train.T @ error_train) / n_train

    #updates the weight
    w = w + step_size * gradient

#compute root-mean-square error on train set after applying LMS
predict_train_lms = phi_train @ w
rms_train_lms = np.sqrt(mean_squared_error(actual_train, predict_train_lms))

#compute root-mean-square error on test set after applying LMS
predict_test_lms = phi_test @ w 
rms_test_lms = np.sqrt(mean_squared_error(actual_test, predict_test_lms))

#prints error for both train and test set
print(f"Root-Mean-Squared-Error on Train Set (LMS): {rms_train_lms:.3f}")
print(f"Root-Mean-Squared-Error on Test Set (LMS): {rms_test_lms:.3f}")