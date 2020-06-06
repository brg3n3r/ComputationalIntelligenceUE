import numpy as np
import matplotlib.pyplot as plt
from svm_plot import plot_decision_function 

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOs. Fill the cost function, the gradient function and gradient descent solver.
"""

def ex_4_a(x, y):
    
    # TODO: Split x, y (take 80% of x, and corresponding y). You can simply use indexing, since the dataset is already shuffled.
    N_train = int(np.size(x,axis=0)*0.8)
    X_train = x[0:N_train]
    y_train = y[0:N_train]#.reshape(N_train,1)
    X_test = x[N_train:]
    y_test = y[N_train:]#.reshape(len(y)-N_train,1)

    C = 1

    # Define the functions of the parameter we want to optimize
    f = lambda th: cost(th, X_train, y_train, C)
    df = lambda th: grad(th, X_train, y_train, C)
    
    # TODO: Initialize w and b to zeros. What is the dimensionality of w?
    w = np.zeros([np.size(X_train, axis=1),1])
    b = 0
    eta = 0.1
    max_iter = 10
    theta_opt, E_list = gradient_descent(f, df, (w, b), eta, max_iter)
    w, b = theta_opt
    
    # TODO: Calculate the predictions using the test set
    y_test_pred = np.ones([len(y)-N_train,1])
    y_test_pred[X_test @ w + b < 0] = -1
    y_test_pred = y_test_pred.reshape(y_test.shape)
    
    # TODO: Calculate the accuracy
    accuracy = list(y_test == y_test_pred).count(True) / np.size(y_test, axis=0)
    print(f'Accuracy: {accuracy}')
    
    # Plot the list of errors
    if len(E_list) > 0:
        fig, ax = plt.subplots(1)
        ax.plot(E_list, linewidth=2)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Error')
        ax.set_title('Error monitoring')
        
    # TODO: Call the function for plotting (plot_decision_function).
    plot_decision_function(theta_opt, X_train, X_test, y_train, y_test)


def gradient_descent(f, df, theta0, learning_rate, max_iter):
    """
    Finds the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decreases the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    """
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm

    E_list = np.zeros(max_iter)
    w, b = theta0

    for ii in range(max_iter):
        E_list[ii] = f((w,b))
        grad_w, grad_b = df((w,b))
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        E_list[ii] = f((w,b))
    # END TODO
    ###########

    return (w, b), E_list


def cost(theta, x, y, C):
    """
    Computes the cost of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: cost
    """
    w = theta[0]
    b = theta[1]
    m = np.size(x, axis=0)
    y = y.reshape(m,1)

    max_res = np.max(np.concatenate((np.zeros([m,1]), 1 - y*(x @ w + b)), axis=1), axis=1)
    cost = 0.5 * np.linalg.norm(w)**2 + C/m * np.sum(max_res, axis=0)

    return cost


def grad(theta, x, y, C):
    """

    Computes the gradient of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: grad_w, grad_b
    """
    w, b = theta
    m = np.size(x, axis=0)
    y = y.reshape(m,1)
    
    I = np.ones([m,1])

    I_decider = (1 - y*(x @ w + b)) < 0
    I[I_decider] = 0

    grad_w = w - C/m * np.sum(I*y*x, axis=0).reshape(w.shape)

    grad_b = - C/m * np.sum(I*y, axis=0)  
    
    return grad_w, grad_b
