import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOS are contained here.
"""


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    clf = svm.SVC(kernel='linear')
    clf.fit(x,y)
    plot_svm_decision_boundary(clf, x, y)

def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    x = np.vstack((x,np.array([4, 0])))
    y = np.hstack((y, np.array([1])))
    clf = svm.SVC(kernel='linear')
    clf.fit(x,y)
    plot_svm_decision_boundary(clf, x, y)


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    x = np.vstack((x,np.array([4, 0])))
    y = np.hstack((y, np.array([1])))

    Cs = [1e6, 1, 0.1, 0.001]
    for element in Cs:
        clf = svm.SVC(C=element, kernel='linear')
        clf.fit(x,y)
        plot_svm_decision_boundary(clf, x, y)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
    print(f'Linear kernel testing score: {clf.score(x_test, y_test)}')

def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the training and test scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 21)
    train_scores = []
    test_scores = []
    test_best = 0

    for element in degrees:
        clf = svm.SVC(kernel='poly', degree=element, coef0=1)
        clf.fit(x_train,y_train)
        train_scores.append(clf.score(x_train, y_train))        
        test_scores.append(clf.score(x_test, y_test))

    plot_score_vs_degree(train_scores, test_scores, degrees)

    test_best = degrees[np.argmax(test_scores)]
    print(f'Polynomial kernel testing score: {np.max(test_scores)}')

    clf = svm.SVC(kernel='poly', degree=test_best, coef0=1)
    clf.fit(x_train,y_train)    

    print(f"Best degree: {test_best}")
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)

def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)

    train_scores = []
    test_scores = []
    test_best = 0

    for element in gammas:
        clf = svm.SVC(kernel='rbf', gamma=element)
        clf.fit(x_train,y_train)
        train_scores.append(clf.score(x_train, y_train))        
        test_scores.append(clf.score(x_test, y_test))

    plot_score_vs_gamma(train_scores, test_scores, gammas)

    test_best = gammas[np.argmax(test_scores)]
    print(f'RBF kernel testing score: {np.max(test_scores)}')

    clf = svm.SVC(kernel='rbf', gamma=test_best)
    clf.fit(x_train,y_train)    

    print(f"Best gamma: {test_best}")
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**5
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Note that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function (parameter baseline)
    ###########

    train_scores = []
    test_scores = []
    train_score_lin = 0
    test_score_lin = 0

    gammas = 10**np.arange(-5, 6, dtype=np.float)    

    clf = svm.SVC(kernel='linear', decision_function_shape='ovr', C=10, )
    clf.fit(x_train,y_train)
    train_score_lin = clf.score(x_train, y_train)        
    test_score_lin = clf.score(x_test, y_test)

    for element in gammas:
        clf = svm.SVC(kernel='rbf', gamma=element, decision_function_shape='ovr', C=10)
        clf.fit(x_train,y_train)
        train_scores.append(clf.score(x_train, y_train))        
        test_scores.append(clf.score(x_test, y_test))

    plot_score_vs_gamma(train_scores, test_scores, gammas, train_score_lin, test_score_lin, baseline=.2)
    


def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 images classified as the most misclassified digit using plot_mnist.
    ###########

    labels = range(1, 6)

    clf = svm.SVC(kernel='linear', decision_function_shape='ovr', C=10)
    clf.fit(x_train,y_train)
    y_test_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_test_pred, labels)
    plot_confusion_matrix(cm, labels)
    print(f'Confusion matrix:\n {cm}')

    i = np.argmin(np.diag(cm))
    sel_err = np.array([y_test != y_test_pred, y_test_pred == labels[i]]).all(axis=0)

    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_test_pred[sel_err], labels=labels[i], k_plots=10, prefix='Predicted class')
