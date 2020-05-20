import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, \
    plot_learned_function, plot_mse_vs_alpha

"""
Assignment 3: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def calculate_mse(nn, x, y):
    """
    Calculates the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    y_pred = nn.predict(x)
    mse = np.linalg.norm(y_pred-y)**2
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    nh = (2,5,50)
    for element in nh:
        nn = MLPRegressor(hidden_layer_sizes=(element,), max_iter=5000, alpha=0, activation='logistic', solver='lbfgs')
        nn.fit(x_train, y_train)
        y_pred_test = nn.predict(x_test)
        plot_learned_function(element, x_train, y_train, [], x_test, y_test, y_pred_test)
        
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    nh = 5
    n_seed = 10
    mse_train = np.zeros([n_seed])
    mse_test = np.zeros([n_seed])
    for index in range(n_seed):
        nn = MLPRegressor(hidden_layer_sizes=(nh,), max_iter=5000, alpha=0, activation='logistic', solver='lbfgs', random_state=index)
        nn.fit(x_train, y_train)
        mse_train[index] = calculate_mse(nn, x_train, y_train)
        mse_test[index] = calculate_mse(nn, x_test, y_test)
        y_pred_test = nn.predict(x_test)
        plot_learned_function(nh, x_train, y_train, [], x_test, y_test, y_pred_test)
        
        
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 d)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass


def ex_1_2(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass
