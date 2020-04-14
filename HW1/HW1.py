#Filename: HW1.py

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits import mplot3d

#--------------------------------------------------------------------------------
# Assignment 1

def main():
    # choose the scenario
    #scenario = 1    # all anchors are Gaussian
    #scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    scenario = 3    # all anchors are exponential
    
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)
    
    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])
                
    plt.figure()       
    plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)
    plt.show()
    
    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)
    
    # get the number of measurements 
    assert(np.size(data,0) == np.size(reference_measurement,0))
    
    #1) ML estimation of model parameters
    params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref)
    
    #2) Position estimation using least squares
    position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, True)
    if scenario == 2:
        # exclude exponential anchor
        position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, False)
        
        
    if(scenario == 3):
        #3) Postion estimation using numerical maximum likelihood
        position_estimation_numerical_ml(data,nr_anchors,p_anchor, params, p_true)
    
        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov, params, p_true)
        
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """

    # plot samples sorted by distance
    dist_ref = np.linalg.norm(p_ref-p_anchor[0,:])
    reference_measurement_sort = np.sort(reference_measurement, axis=0)
    nr_samples = np.size(reference_measurement, 0)
    x = range(0, nr_samples)
    
    plt.figure()
    plt.plot(x, dist_ref * np.ones(nr_samples))
    plt.plot(x, reference_measurement_sort)
    plt.legend(["True Distance","Measurement 1","Measurement 2","Measurement 3","Measurement 4"])
    plt.xlabel("samples")
    plt.ylabel("distance/m")
    plt.grid(True)
    plt.show()
    
    # estimate parameters
    params = np.zeros([1, nr_anchors])
    for ii in range(0, nr_anchors):
        dist_error = reference_measurement[:,ii] - dist_ref
        # check whether anchor is Gaussian or exponential
        if reference_measurement_sort[0,ii] >= dist_ref:
            # exponential
            params[0,ii] = nr_samples / np.sum(dist_error)
        else:
            # Gaussian
            params[0,ii] = np.sum(dist_error**2) / nr_samples
    
    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, use_exponential):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    
    # set parameters
    tol = 10**(-4)  # tolerance
    max_iter = 100  # maximum iterations for GN
    p_start = np.random.uniform([-5, 5])
    nr_samples = np.size(data,0)
    
    # adjust for fewer anchors
    if use_exponential == False:
        data = data[:,1:4]
        p_anchor = p_anchor[1:4,:]
        nr_anchors = 3
       
    # estimate position for all samples
    p_ls = np.zeros([nr_samples,2])
    for ii in range(0, nr_samples):
        p_ls[ii,:] = least_squares_GN(p_anchor,p_start, data[ii,:], max_iter, tol)
        
	# calculate error measures and create plots----------------
    error_ls = np.linalg.norm(p_ls - p_true, axis=1)
    var_error_ls = np.var(error_ls)
    print("LS Error Variance = ", var_error_ls)
    mu_error_ls = np.mean(error_ls)
    print("LS Error Mean = ", mu_error_ls)
    
    mu_p_ls = np.mean(p_ls, axis = 0)
    cov_p_ls = np.cov(p_ls, rowvar = False)
    
    # Cumulative Distribution Function
    plt.figure("ecdf")
    Fx, x = ecdf(error_ls)
    plt.plot(x, Fx)
    plt.xlabel("error/m")
    plt.ylabel("Empirical Cumulative Distribution Function")
    plt.grid(True)
    #plt.legend("Least-Squares")
    plt.show()
  
    
    # scatter plots
    plt.figure()
    plot_anchors_and_agent(nr_anchors, p_anchor, p_true)
    plt.scatter(p_ls[:,0], p_ls[:,1], s=0.5)
    plot_gauss_contour(mu_p_ls, cov_p_ls, -6, 6, -6, 6)
    plt.show()
    
    plt.figure()
    plt.axis([0, 4, -6, -2])
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.1, p_true[0, 1] + 0.1, r'$p_{true}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.scatter(p_ls[:,0], p_ls[:,1], s=1)
    plot_gauss_contour(mu_p_ls, cov_p_ls, 0, 4, -6, -2)
    plt.show
    
    pass
#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    # initialize variables
    nr_samples = np.size(data,0)
    x = np.linspace(-5,5,201)
    y = np.linspace(5,-5,201)    
    likelihood_grid = np.zeros([nr_samples,201,201]) # each slice contains the grid for one sample
    
    # go through grid    
    for jj in range(0, 201):
        for ii in range(0,201):
            # compute current distance to all anchors
            dist_p = np.linalg.norm(p_anchor - [x[ii],y[jj]], axis=1)
            # find slices where (measured distance >= current distance) true for ALL anchors
            nonzero_likelihood_idx = (data >= dist_p).all(axis=1)
            # compute joint likelihood for those slices only
            likelihood_grid[nonzero_likelihood_idx,jj,ii] = np.prod(lambdas*np.exp(-lambdas*(data[nonzero_likelihood_idx,:]-dist_p)),axis=1)
            
    # plot grid of first sample
    X, Y = np.meshgrid(x, y)
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X,Y,likelihood_grid[0,:,:],cmap = "winter")
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    ax.set_zlabel("Joint Likelihood")
    ax.ticklabel_format(scilimits = (-1,0))
    plt.show()                
    
    # find maximum/position estimate for all samples
    max_idx = likelihood_grid.reshape(likelihood_grid.shape[0],-1).argmax(1)
    max_ji = np.column_stack(np.unravel_index(max_idx, likelihood_grid[0,:,:].shape))
    p_nml = np.column_stack((x[max_ji[:,1]],y[max_ji[:,0]]))
    
    # calculate error measures and create plots----------------
    error_nml = np.linalg.norm(p_nml - p_true, axis=1)
    var_error_nml = np.var(error_nml)
    print("NML Error Variance = ", var_error_nml)
    mu_error_nml = np.mean(error_nml)
    print("NML Error Mean = ", mu_error_nml)
    
    mu_p_nml = np.mean(p_nml, axis = 0)
    cov_p_nml = np.cov(p_nml, rowvar = False)
    
    # Cumulative Distribution Function
    plt.figure("ecdf")
    Fx, x = ecdf(error_nml)
    plt.plot(x, Fx)
    plt.xlabel("error/m")
    plt.ylabel("Empirical Cumulative Distribution Function")
    plt.grid(True)
    plt.show()
    
    # scatter plots
    plt.figure()
    plot_anchors_and_agent(nr_anchors, p_anchor, p_true)
    plot_gauss_contour(mu_p_nml, cov_p_nml, -6, 6, -6, 6)
    plt.scatter(p_nml[:,0], p_nml[:,1], s=0.5)
    plt.show()
    
    # Gauss Contour
    plt.figure()
    plt.axis([0, 4, -6, -2])
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.1, p_true[0, 1] + 0.1, r'$p_{true}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.scatter(p_nml[:,0], p_nml[:,1], s=1)
    plot_gauss_contour(mu_p_nml, cov_p_nml, -6, 6, -6, 6)
    plt.show
    
    pass
#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """
         
    nr_samples = np.size(data,0)
    x = np.linspace(-5,5,201)
    y = np.linspace(5,-5,201)    
    likelihood_grid = np.zeros([nr_samples,201,201])
        
    for jj in range(0, 201):
        for ii in range(0,201):
            # current position
            p = np.array([x[ii],y[jj]])
            # compute current distance to all anchors
            dist_p = np.linalg.norm(p_anchor - p, axis=1)
            # compute prior for current position
            prior = get_prior(p, prior_mean, prior_cov)
            # find slices where (measured distance >= current distance) true for ALL anchors
            nonzero_likelihood_idx = (data >= dist_p).all(axis=1)
            # compute joint likelihood (including prior) for those slices only
            likelihood_grid[nonzero_likelihood_idx,jj,ii] = np.prod(lambdas*np.exp(-lambdas*(data[nonzero_likelihood_idx,:]-dist_p))*prior,axis=1)
    
    # plot grid of first sample        
    # X, Y = np.meshgrid(x, y)
    # plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.plot_surface(X,Y,likelihood_grid[0,:,:],cmap = "winter")
    # plt.xlabel("x/m")
    # plt.ylabel("y/m")
    # ax.set_zlabel("Joint Likelihood")
    # plt.show()                
    
    # find maximum/position estimate for all samples
    max_idx = likelihood_grid.reshape(likelihood_grid.shape[0],-1).argmax(1)
    max_ji = np.column_stack(np.unravel_index(max_idx, likelihood_grid[0,:,:].shape))
    p_bayes = np.column_stack((x[max_ji[:,1]],y[max_ji[:,0]]))
    
    # calculate error measures and create plots----------------
    error_bayes = np.linalg.norm(p_bayes - p_true, axis=1)
    var_error_bayes = np.var(error_bayes)
    print("Bayes Error Variance = ", var_error_bayes)
    mu_error_bayes = np.mean(error_bayes)
    print("Bayes Error Mean = ", mu_error_bayes)
    
    mu_p_bayes = np.mean(p_bayes, axis = 0)
    cov_p_bayes = np.cov(p_bayes, rowvar = False)
    
    # Cumulative Distribution Function
    plt.figure("ecdf")
    Fx, x = ecdf(error_bayes)
    plt.plot(x, Fx)
    plt.xlabel("error/m")
    plt.ylabel("Empirical Cumulative Distribution Function")
    plt.legend(["Least-Squares","Numerical Maximum Likelihood","Bayes"])
    plt.grid(True)
    plt.show()
    
    # scatter plots
    plt.figure()
    plot_anchors_and_agent(nr_anchors, p_anchor, p_true)
    plot_gauss_contour(mu_p_bayes, cov_p_bayes, -6, 6, -6, 6)
    plt.scatter(p_bayes[:,0], p_bayes[:,1], s=0.5)
    plt.show()
    
    # Gauss Contour
    plt.figure()
    plt.axis([0, 4, -6, -2])
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.1, p_true[0, 1] + 0.1, r'$p_{true}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.scatter(p_bayes[:,0], p_bayes[:,1], s=1)
    plot_gauss_contour(mu_p_bayes, cov_p_bayes, -6, 6, -6, 6)
    plt.show
    
    pass
#--------------------------------------------------------------------------------
def get_prior(p, prior_mean, prior_cov):
    """ calculate prior gaussian probability for current grid position
    Input:
        p... current grid position, 1x2
        prior_mean... mean of prior distribution, 1x2
        prior_cov... covariance of prior distribution, 2x2"""
    
    prior = 1/(2*np.pi*np.linalg.det(prior_cov)**0.5) * np.exp(-0.5*(((p-prior_mean).dot(prior_cov)).dot(np.transpose(p-prior_mean))))
        
    return prior
#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor,p_start, measurements_n, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        measurements_n... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""
    
    p_t = p_start
    for ii in range(0, max_iter):    
        dist_t = np.linalg.norm(p_anchor - p_t, axis=1)
        
        # compute jacobian matrix
        J = get_jacobian_matrix(p_anchor, p_t, dist_t)
        
        # perform algorithm for current iteration
        JJ_inv = np.linalg.inv((np.transpose(J)).dot(J))
        JJ_inv_J = JJ_inv.dot(np.transpose(J))
        p_next = p_t - (JJ_inv_J.dot(measurements_n - dist_t))
        
        # check for break condition
        est_change = np.linalg.norm(p_next - p_t)
        if est_change < tol:
            break
        
        # prepare for next iteration
        p_t = p_next
        
    return p_next
#--------------------------------------------------------------------------------
def get_jacobian_matrix(p_anchor, p_t, dist_t):
    """ build Jacobian Matrix for all anchors based on previous point estimation
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_curr... current position estimate, 1x2
        dist_t... distance of anchors to current point estimate, nr_anchors x 1"""
    
    J = np.zeros([np.size(p_anchor,0), 2])
    J[:,0] = (p_anchor[:,0] - p_t[0]) / dist_t
    J[:,1] = (p_anchor[:,1] - p_t[1]) / dist_t
        
    return J
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,title="Title"):
    
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""
    
	#npts = 100
    delta = 0.025
    X, Y = np.mgrid[xmin:xmax:delta, ymin:ymax:delta]
    pos = np.dstack((X, Y))
                    
    Z = stats.multivariate_normal(mu, cov)
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    plt.gca().set_aspect("equal")
    CS = plt.contour(X, Y, Z.pdf(pos),3,colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    #plt.title(title)
    #plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):   
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'
    
    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)
    
    return (data,reference)
#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    #plt.show()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    plt.close('all')
    main()