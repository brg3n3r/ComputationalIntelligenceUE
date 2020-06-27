#Filename: HW5_skeleton.py
#Author: Christian Knoll
#Edited: May 2020

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets

#--------------------------------------------------------------------------------
# Assignment 5
def main():
    #------------------------
    # 0) Get the input
    ## (a) load the modified iris data
    data, labels, feature_names = load_iris_data()

    ## (b) construct the datasets
    x_2dim = data[:,[0,2]]
    x_4dim = data

    #TODO: implement PCA
    x_2dim_pca = PCA(data,nr_dimensions=2,whitening=False)

    ## (c) visually inspect the data with the provided function (see example below)
    plt.figure()
    plot_iris_data(x_2dim,labels, feature_names[0], feature_names[2], "Iris Dataset")
    plt.show()

    algorithm = 1 #1: EM, 2: K-means
    scenario = 1 #1: 2 features, 2: 4 features
    diagonal = False
    
    legend_cluster = ["Cluster 1", "Cluster 1 - mean", "Cluster 2", "Cluster 2 - mean", "Cluster 3", "Cluster 3 - mean", "Cluster 4", "Cluster 4 - mean"]
    legend_labels = ["Iris-Setosa","Iris-Setosa - mean","Iris-Versicolor","Iris-Versicolor - mean","Iris-Virginica","Iris-Virginica - mean"]

    if scenario == 1:
        #------------------------
        # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm
        dim = 2
    
        # set parameters
        tol = 0.1  # tolerance
        max_iter = 50  # maximum iterations for GN    
        nr_components_list = range(2,5)
    
        for nr_components in nr_components_list:
            
            if algorithm == 1:
                alpha_0, mean_0, cov_0 = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario, X=x_2dim)
                alpha, mean, cov, log_likelihood, labels_pred = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol, feature_names)
            
                plt.figure()
                for component in range(nr_components):
                    plt.scatter(x_2dim[component==labels_pred,0],x_2dim[component==labels_pred,1],color='C'+str(component))
                    plot_gauss_contour(mean[:,component], cov[:,:,component], 4.1, 8.1, 0.7, 7.2, np.size(data,axis=0), cluster=component)
                if nr_components == 3:
                    plt.legend((legend_labels[::2]))
                else:
                    plt.legend((legend_cluster[:2*nr_components:2]))
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[2])
                plt.title(f'EM with K={nr_components}')
                plt.show()
                
                plt.figure()
                plt.plot(log_likelihood)
                plt.xlabel('iterations')
                plt.ylabel('log-likelihood')
                plt.title(f'EM with K={nr_components}')
                plt.show()
                #print(log_likelihood[-1])
        
            elif algorithm == 2:
                initial_centers = init_k_means(dimension = dim, nr_clusters=nr_components, scenario=scenario, X=x_2dim)
                centers, cumulative_distance, labels_pred = k_means(x_2dim, nr_components, initial_centers, max_iter, tol, feature_names)
                
                plt.figure()
                for cluster in range(nr_components):
                    plt.scatter(x_2dim[cluster==labels_pred,0], x_2dim[cluster==labels_pred,1], facecolors='none', edgecolors='C'+str(cluster), marker='o')
                    plt.scatter(centers[0,cluster], centers[1,cluster], c='C'+str(cluster), marker='o', s=50)
                if nr_components == 3:
                    plt.legend((legend_labels))
                else:
                    plt.legend((legend_cluster[:2*nr_components]))
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[2])
                plt.title(f'K-means with K={nr_components}')
                plt.show()
                
                plt.figure()
                plt.plot(cumulative_distance)
                plt.xlabel('iterations')
                plt.ylabel('cumulative distance')
                plt.title(f'K-means with K={nr_components}')
                plt.show()
                #print(cumulative_distance[-1])
            
            if nr_components == 3:
                accuracy = score(labels, labels_pred)
                print(f'Accuracy (K=3): {accuracy*100}%')
                
        
                
    
    elif scenario == 2: 
        #------------------------
        # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm
        dim = 4
        
        tol = 0.1  # tolerance
        max_iter = 50  # maximum iterations for GN    
        nr_components_list = range(2,5)
    
        for nr_components in nr_components_list:
            
            if algorithm == 1:
                alpha_0, mean_0, cov_0 = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario, X=x_4dim)
                alpha, mean, cov, log_likelihood, labels_pred = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol, feature_names, diagonal)
                    
                plt.figure()
                for component in range(nr_components):
                    plt.scatter(x_4dim[component==labels_pred,0],x_4dim[component==labels_pred,2],color='C'+str(component))
                    plot_gauss_contour(mean[[0,2],component], cov[[0,2],[0,2],component], 4.1, 8.1, 0.7, 7.2, np.size(data, axis=0),cluster=component)
                if nr_components == 3:
                    plt.legend((legend_labels[::2]))
                else:
                    plt.legend((legend_cluster[:2*nr_components:2]))
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[2])
                plt.title(f'EM with K={nr_components}')
                plt.show()
                
                plt.figure()
                plt.plot(log_likelihood)
                plt.xlabel('iterations')
                plt.ylabel('log-likelihood')
                plt.title(f'EM with K={nr_components}')
                plt.show()
                #print(log_likelihood[-1])
        
            elif algorithm == 2:
                initial_centers = init_k_means(dimension = dim, nr_clusters=nr_components, scenario=scenario, X=x_4dim)
                centers, cumulative_distance, labels_pred = k_means(x_4dim, nr_components, initial_centers, max_iter, tol, feature_names)
                                
                plt.figure()
                for cluster in range(nr_components):
                    plt.scatter(x_4dim[cluster==labels_pred,0], x_4dim[cluster==labels_pred,2], facecolors='none', edgecolors='C'+str(cluster), marker='o')
                    plt.scatter(centers[0,cluster], centers[2,cluster], c='C'+str(cluster), marker='o', s=50)
                if nr_components == 3:
                    plt.legend((legend_labels))
                else:
                    plt.legend((legend_cluster[:2*nr_components]))
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[2])
                plt.title(f'K-means with K={nr_components}')
                plt.show()
                
                plt.figure()
                plt.plot(cumulative_distance)
                plt.xlabel('iterations')
                plt.ylabel('cumulative distance')
                plt.title(f'K-means with K={nr_components}')
                plt.show()
                #print(cumulative_distance[-1])
                
            if nr_components == 3:
                accuracy = score(labels, labels_pred)
                print(f'Accuracy (K=3): {accuracy*100}%')


    #------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data
    scenario = 3
    dim = 2
    nr_components = 3

    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter =   # maximum iterations for GN
    #nr_components = ... #n number of components

    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_2dim_pca, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    #initial_centers = init_k_means(dimension = dim, nr_cluster=nr_components, scenario=scenario)
    #... = k_means(x_2dim_pca, nr_components, initial_centers, max_iter, tol)

    #TODO: visualize your results
    #TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)

    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
    
def score(y, y_pred):
    """ computes accuracy achieved on classification
    Input:
        y... true labels, nr_samples x 1
        y_pred... predicted labels, nr_samples x 1
    Returns:
        score... accuracy"""
        
    score = np.sum(y == y_pred) / y.shape[0]
    
    return score

def rearrange_labels(y_pred, order):
    """ assigns correct labels to classification results
    Input:
        y_pred... predicted labels, nr_samples x 1
        order... label translation table, 3 x 1
    Output:
        y_pred_new... translated predicted labels, nr_samples x 1
        new_order... order in which the labels were rearranged, 3 x 1"""
        
    y_pred_new = np.zeros((y_pred.shape))
    new_order = np.empty((1,0))
    for class_name in range(3):
        y_pred_new[y_pred == class_name] = order[class_name]
        new_order = np.concatenate((new_order,np.array(np.where(order==class_name))),axis=1)
    new_order = new_order.astype(int).reshape(3)
    
    return y_pred_new, new_order

def init_EM(dimension=2,nr_components=3, scenario=None, X=None):
    """ initializes the EM algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... (optional) parameter that allows to further specify the settings, scalar
        X... (optional) samples that may be used for proper inititalization, nr_samples x dimension(D)
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""
    
    #alpha_0 = np.random.rand(1, nr_components)
    #alpha_0 = alpha_0 / np.sum(alpha_0, axis=1)
    alpha_0 = np.ones((1,nr_components)) / nr_components
    
    nr_samples = np.size(X, axis=0)
    rand_samples = np.random.randint(0, nr_samples, size=nr_components)
    if nr_components == 3:
        #good starting samples:
        #2Dim:
        #rand_samples = [8, 69, 136]
        #rand_samples = [67, 73, 123]
        
        #4Dim:
        #rand_samples = [115  60  34]
        #rand_samples = [81 58 67]
        #rand_samples = [ 91 132 136]
        
        print(f'Samples used for initial mean (K=3): {rand_samples}')
    mean_0 = X[rand_samples,:].T
    
    cov_0 = np.empty((dimension, dimension, nr_components))
    for component in range(nr_components):
        cov_0[:,:,component] = np.cov(X, rowvar=False)
        
    return alpha_0, mean_0, cov_0

#--------------------------------------------------------------------------------
def EM(X,K,alpha_0,mean_0,cov_0, max_iter, tol, feature_names, diagonal=False):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension
    D = X.shape[1]
    assert D == mean_0.shape[0]
    
    if D == 2:
        plot_dim = [0,1]
    elif D == 4:
        plot_dim = [0,2]
    
    nr_samples = np.size(X, axis=0)
    
    mean = mean_0
    cov = cov_0
    if diagonal == True:
        for component in range(K): 
            cov[:,:,component] = np.diag(np.diag(cov[:,:,component]))
    alpha = alpha_0
    log_likelihood = np.zeros((1))

    for iteration in range(max_iter):
        r = np.empty((0, nr_samples))
        for component in range(K):
            r = np.concatenate((r, (alpha[:,component] * likelihood_multivariate_normal(
                X, mean[:,component], cov[:,:,component])).reshape(1, nr_samples)), axis=0)
        r = r / np.sum(r, axis=0)
        
        labels = np.argmax(r, axis=0)
        
        if K == 3 and iteration%2 == 0:
            label_order = reassign_class_labels(labels)
            labels_sorted, new_order = rearrange_labels(labels, label_order)
            
            plt.figure()
            plot_iris_data(X[:,plot_dim], labels_sorted, feature_names[0], feature_names[2], f'EM with K={K} at iteration {iteration}')
            plt.show()
        
        mean = (1/np.sum(r, axis=1).reshape(K,1) * r@X).T
        
        for component in range(K):
            cov[:,:,component] = 1/np.sum(r[component,:]) * (r[component,:].reshape(nr_samples,1)*(X-mean[:,component])).T @ (X-mean[:,component])
            if diagonal == True:
                cov[:,:,component] = np.diag(np.diag(cov[:,:,component]))
        
        alpha = (np.sum(r, axis=1) / nr_samples).reshape(1,K)
        
        likelihood_current = np.empty((0, nr_samples))
        for component in range(K):
            likelihood_current = np.concatenate((likelihood_current, (alpha[:,component] * likelihood_multivariate_normal(
                X, mean[:,component], cov[:,:,component])).reshape(1, nr_samples)), axis=0)
        log_likelihood = np.concatenate((log_likelihood, np.sum(np.log(np.sum(likelihood_current, axis=0).reshape(1,nr_samples)), axis=1)), axis=0)
        
        log_likelihood_diff = np.diff(log_likelihood, axis=0)
        if abs(log_likelihood_diff[iteration]) <= tol:
            
            break

    if K == 3:
        label_order = reassign_class_labels(labels)
        labels, new_order = rearrange_labels(labels, label_order)
        mean = mean[:,new_order]
        cov = cov[:,:,new_order]
        
        plt.figure()
        plot_iris_data(X[:,plot_dim], labels, feature_names[0], feature_names[2], f'EM with K={K} after convergence')
        plt.show()        
    
    return alpha, mean, cov, log_likelihood[1:], labels
#--------------------------------------------------------------------------------
def init_k_means(dimension=None, nr_clusters=None, scenario=None, X=None):
    """ initializes the k_means algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... (optional) parameter that allows to further specify the settings, scalar
        X... (optional) samples that may be used for proper inititalization, nr_samples x dimension(D)
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    #TODO: chosse suitable inital values for each scenario
    nr_samples = np.size(X, axis=0)
    rand_samples = np.random.randint(0, nr_samples, size=nr_clusters)
    if nr_clusters == 3:
        #rand_samples = [8, 69, 136]
        print(f'Samples used for initial centers (K=3): {rand_samples}')
    initial_centers = X[rand_samples,:].T
    
    return initial_centers
#--------------------------------------------------------------------------------
def k_means(X,K, centers_0, max_iter, tol, feature_names):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    assert D == centers_0.shape[0]
    
    if D == 2:
        plot_dim = [0,1]
    elif D == 4:
        plot_dim = [0,2]
    
    nr_samples = np.size(X,axis=0)
    centers = centers_0
    cumulative_distance = np.zeros((1))
    
    for iteration in range(max_iter):
        
        distance = np.empty((nr_samples,0))
        for cluster in range(K):
            distance = np.concatenate((distance, np.sum((X-centers[:,cluster].T)**2, axis=1).reshape(nr_samples,1)), axis=1)
        labels = np.argmin(distance, axis=1)
        
        if K == 3:
            label_order = reassign_class_labels(labels)
            labels_sorted, new_order = rearrange_labels(labels, label_order)
            
            plt.figure()
            plot_iris_data(X[:,plot_dim], labels_sorted, feature_names[0], feature_names[2], f'K-means with K={K} at iteration {iteration}')
            plt.show()
        
        cluster_distance = np.empty((1,0))
        for cluster in range(K):
            centers[:,cluster] = np.mean(X[labels == cluster, :], axis=0)#.reshape(D,1)
            cluster_distance = np.concatenate((cluster_distance, np.sum(np.sum((X[labels==cluster,:]-centers[:,cluster].T)**2, axis=1), axis=0).reshape(1,1)), axis=1)

        cumulative_distance = np.concatenate((cumulative_distance, np.sum(cluster_distance).reshape(1)), axis=0)
        
        cumulative_distance_diff = np.diff(cumulative_distance, axis=0)
        if abs(cumulative_distance_diff[iteration]) <= tol:
            distance = np.empty((nr_samples,0))
            for cluster in range(K):
                distance = np.concatenate((distance, np.sum((X-centers[:,cluster].T)**2, axis=1).reshape(nr_samples,1)), axis=1)
            labels = np.argmin(distance, axis=1)
            break
        
    if K == 3:
        label_order = reassign_class_labels(labels)
        labels, new_order = rearrange_labels(labels, label_order)
        centers = centers[:,new_order]
        
        plt.figure()
        plot_iris_data(X[:,plot_dim], labels, feature_names[0], feature_names[2], f'K-means with K={K} after convergence')
        plt.show()
    
    return centers, cumulative_distance[1:], labels
#--------------------------------------------------------------------------------
def PCA(data,nr_dimensions=None, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening

    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    #TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components


    #TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input:
    Returns:
        X... samples, 150x4
        Y... labels, 150x1
        feature_names... name of the data columns"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100,2] =  iris.data[50:100,2]-0.25
    Y = iris.target
    return X,Y, iris.feature_names
#--------------------------------------------------------------------------------
def plot_iris_data(data, labels, x_axis, y_axis, title):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1
        x_axis... label for the x_axis
        y_axis... label for the y_axis
        title...  title of the plot"""

    plt.scatter(data[labels==0,0], data[labels==0,1], label='Iris-Setosa')
    plt.scatter(data[labels==1,0], data[labels==1,1], label='Iris-Versicolor')
    plt.scatter(data[labels==2,0], data[labels==2,1], label='Iris-Virginica')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    #plt.show()
#--------------------------------------------------------------------------------
def likelihood_multivariate_normal(X, mean, cov, log=False):
   """Returns the likelihood of X for multivariate (d-dimensional) Gaussian
   specified with mu and cov.

   X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
   mean ... mean -- [mu_1, mu_2,...,mu_d]
   cov ... covariance matrix -- np.array with (d x d)
   log ... False for likelihood, true for log-likelihood
   """

   dist = multivariate_normal(mean, cov)
   if log is False:
       P = dist.pdf(X)
   elif log is True:
       P = dist.logpdf(X)
   return P

#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,nr_points,title="Title",cluster=0):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters

    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""

	#npts = 100
    delta_x = float(xmax-xmin) / float(nr_points)
    delta_y = float(ymax-ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)


    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = multivariate_normal(mu, cov).pdf(pos)
    plt.plot([mu[0]],[mu[1]],marker='+',color='C'+str(cluster)) # plot the mean as a single point
    CS = plt.contour(X, Y, Z, colors='C'+str(cluster))
    plt.clabel(CS, inline=1, fontsize=10)
    #plt.show()
    return
#--------------------------------------------------------------------------------
def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.

    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)

    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return np.random.permutation(y) # permutation of all samples
#--------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable.
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50]==0)   ,  np.sum(labels[0:50]==1)   , np.sum(labels[0:50]==2)   ],
                                  [np.sum(labels[50:100]==0) ,  np.sum(labels[50:100]==1) , np.sum(labels[50:100]==2) ],
                                  [np.sum(labels[100:150]==0),  np.sum(labels[100:150]==1), np.sum(labels[100:150]==2)]])
    new_labels = np.array([np.argmax(class_assignments[:,0]),
                           np.argmax(class_assignments[:,1]),
                           np.argmax(class_assignments[:,2])])

    return new_labels
#--------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)

    plot_gauss_contour(mu, cov, -2, 2, -2, 2,100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)

    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))

    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels =np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd==0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd==1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd==2] = new_labels[2]

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    sanity_checks()
    plt.rcParams['figure.max_open_warning'] = 0
    plt.close('all')
    main()
