#!/usr/bin/env python

import numpy as np
import json
import matplotlib.pyplot as plt
from plot_poly import plot_poly, plot_errors
import poly
#!!! import os

"""
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file:
1) loads the data from 'data_linreg.json'
2) trains and tests a linear regression model for K degrees
3) TODO: select the degree that minimizes validation error
4) plots the optimal results

TODO boxes are here and in 'poly.py'
"""


def main():
    # Number of possible degrees to be tested
    K = 30
    #files = os.getcwd()
    #print(files)
    #os.chdir('C:/Users/mbuergener/Desktop/CI_Temp/CI_HW2/code/linear_regression') #!!!!!!!!!!!!!!!!!
    data_path = 'data_linreg.json'

    # Load the data
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    # Init vectors storing MSE (Mean squared error) values of each set at each degrees
    mse_train = np.zeros(K)
    mse_val = np.zeros(K)
    mse_test = np.zeros(K)
    theta_list = np.zeros(K, dtype=object)
    degrees = np.arange(K) + 1

    # Compute the MSE values
    for i in range(K):
        theta_list[i], mse_train[i], mse_val[i], mse_test[i] = poly.train_and_test(data, degrees[i])

    ######################
    #
    # TODO
    #
    # Find the best degree that minimizes the validation error.
    # Store it in the variable i_best for plotting the results
    #
    # TIPs:
    # - use the argmin function of numpy
    # - the code above is already giving the vectors of errors
    i_best_train = np.argmin(mse_train)
    i_best_val = np.argmin(mse_val)
    #i_best_test = np.argmin(mse_test)

    #mse_train_norm = mse_train / np.max(mse_train) 
    #mse_val_norm = mse_val / np.max(mse_val) 
    #mse_test_norm = mse_test / np.max(mse_test) 

    #
    # END TODO
    ######################

    i_plots = np.array([1, 5, 10, 22]) - 1
    i_plots = np.append(i_plots,[i_best_train, i_best_val])

    #Plot the training error as a function of the degrees
    #plt.figure()
    #plot_errors(i_best, degrees, mse_train, mse_val, mse_test)
    #plot_poly(data, best_degree, best_theta)
    #plt.show()
    
    for element in i_plots:
        plot_poly(data, degrees[element], theta_list[element])
        plt.tight_layout()
        plt.show()

    #plot_errors(i_best_test, degrees, mse_train_norm, mse_val_norm, mse_test_norm)
    #plt.show()
    
    plt.figure() #!!!
    plot_errors(i_best_val, degrees, mse_train, mse_val, mse_test)
    plt.show()

    

if __name__ == '__main__':
    plt.close('all')
    main()
