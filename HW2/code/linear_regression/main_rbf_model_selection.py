#!/usr/bin/env python
import numpy as np
import json
import matplotlib.pyplot as plt
from plot_rbf import plot_rbf, plot_errors
import rbf
#!!! import os

"""
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file:
1) loads the data from 'data_linreg.json'
2) trains and tests a linear regression model for K different numbers of RBF centers
3) TODO: select the best cluster center number
3) plots the optimal results

TODO boxes are to be found here and in 'rbf.py'
"""


def main():
    # Number of possible degrees to be tested
    K = 40
    data_path = 'data_linreg.json'

    # Load the data
    #os.chdir('C:/Users/mbuergener/Desktop/CI_Temp/CI_HW2/code/linear_regression') #!!!!!!!!!!!!!!!!
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    ######################
    #
    # TODO
    #
    # Compute the arrays containing the Mean Squared Errors in all cases
    #
    # Find the degree that minimizes the validation error
    # Store it in the variable i_best for plotting the results
    #
    # TIP:
    # - You are invited to adapt the code you did for the polynomial case
    # - use the argmin function of numpy
    #

    # Init vectors storing MSE (Mean squared error) values of each set at each degrees
    mse_train = np.zeros(K)
    mse_val = np.zeros(K)
    mse_test = np.zeros(K)
    theta_list = np.zeros(K, dtype=object)
    n_centers = np.arange(K) + 1

    # Compute the MSE values
    #i_best = 0

    for i in range(K):
        theta_list[i], mse_train[i], mse_val[i], mse_test[i] = rbf.train_and_test(data, n_centers[i])

    i_best_train = np.argmin(mse_train)
    i_best_val = np.argmin(mse_val)
    
    i_plots = np.array([1, 5, 10, 22]) - 1
    i_plots = np.append(i_plots,[i_best_train, i_best_val])

    for element in i_plots:
        plot_rbf(data, n_centers[element], theta_list[element])
        plt.tight_layout()
        #plt.savefig('linreg_rbf1_ncent'+str(element+1)+'.pdf',format='pdf')
        plt.show()
    
    #
    # TODO END
    ######################

    # Plot the training error as a function of the degrees
    plt.figure()
    plot_errors(i_best_val, n_centers, mse_train, mse_val, mse_test)
    #plt.savefig("linreg_rbf4_error.pdf",format='pdf')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    main()
