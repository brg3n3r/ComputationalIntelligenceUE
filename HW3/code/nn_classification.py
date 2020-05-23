from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_boxplot, plot_image
import numpy as np
import pickle


"""
Assignment 3: Neural networks
Part 2: Classification with Neural Networks: Fashion MNIST

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""

IMAGE_DIM = (28, 28)

def ex_2_1(X_train, y_train, X_test, y_test):
    """
    Solution for exercise 2.1
    :param X_train: Train set
    :param y_train: Targets for the train set
    :param X_test: Test set
    :param y_test: Targets for the test set
    :return:
    """
    
    nh = 100
    # n_seeds = 5
    # train_acc = np.zeros([n_seeds])
    # test_acc = np.zeros([n_seeds])
    # for index in range(n_seeds):
    #     nn = MLPClassifier(hidden_layer_sizes=(nh,), activation='tanh', max_iter=50, random_state=index)
    #     nn.fit(X_train, y_train)
    #     train_acc[index] = nn.score(X_train, y_train)
    #     test_acc[index] = nn.score(X_test, y_test)
    
    # plot_boxplot(train_acc, test_acc)
    # best_seed = np.argmax(test_acc)
    # print(best_seed)
    best_seed = 2

    #nn = MLPClassifier(hidden_layer_sizes=(nh,), activation='tanh', max_iter=50, random_state=best_seed)

    #nn.fit(X_train, y_train)

    #pickle.dump(nn, open('nn100.sav', 'wb'))
    nn = pickle.load(open('C:/Users/mbuergener/Desktop/Temporary/CI/HW3/code/nn100.sav', 'rb'))  

    y_test_pred = nn.predict(X_test)

    print(confusion_matrix(y_test, y_test_pred))

    coefs = nn.coefs_

    plot_hidden_layer_weights(nn.coefs_[0])

    plot_vct = X_test[y_test != y_test_pred]

    for ii in range(5):
        plot_image(plot_vct[ii].reshape(IMAGE_DIM))

    
