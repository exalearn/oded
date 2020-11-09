import numpy as np
from scipy.stats import multivariate_normal

def compute_expectation_with_random_sampling(func,x_samples):
    """
    Compute E[func] \approx (1/N) * sum( func(x_samples) )

    Inputs

    -------
    func : function that takes n_params arguments 
    x_samples : list(n_params), where each element is a list(n_samples)

    Outputs
    
    -------
    e_func : double
        average of func over n_samples
    """
    n_samples = len(x_samples[0])
    e_func    = (1./n_samples) * np.sum( func(*x_samples) )
    return e_func

def compute_total_probabilities(rho_y_given_theta,rho_theta):
    """
    Compute rho(y_j) = sum_i rho(y_j|theta_i) * rho(theta_i)

    Inputs

    -------
    rho_y_given_theta : numpy.ndarray([n_theta,n_y])
        Discrete conditional probabilities of y_j given theta_i
    rho_theta : numpy.ndarray(n_theta)
        Discrete probabilities of theta_i

    Outputs
    
    -------
    rho_y : numpy.ndarray(n_y)
        Discrete probabilities of y_j
    """
    n_theta,n_y = np.shape(rho_y_given_theta)
    assert( len(rho_theta) == n_theta )
    assert( np.all( np.isclose( np.sum(rho_y_given_theta,axis=1) , 1) ) )
    assert( np.isclose( np.sum(rho_theta) , 1 ) )
    rho_y = np.sum( rho_y_given_theta * np.tile(rho_theta,[n_y,1]).T , axis=0 )
    return rho_y

def compute_weighted_average(x,rho):
    """
    Compute sum_j rho(j) * x_j

    Inputs
    
    -------
    x : list(n_x) of numpy.array(d_x)
        Discrete set of n_x values
    rho : numpy.narray(n_x)
        Discrete probabilities of n_x values of x

    Outputs

    -------
    avg : double
        Weighted average
    """
    assert( len(x) == len(rho) )
    #assert( np.isclose(np.sum(rho) , 1) )
    avg = np.sum( rho*x )
    return avg
    
def compute_probability_theta_given_y( rho_y_given_theta , rho_theta ):
    """
    Computes rho_theta_given_y = rho_y_given_theta * rho_theta / rho_y

    Inputs

    -------
    rho_y_given_theta : numpy.ndarray([n_theta,n_y])
        Discrete conditional probabilities, where entry (i,j) = rho(y_j | theta_i)
    rho_theta : numpy.array(n_theta) 
        Discrete probability mass function for n_theta values of theta

    Outputs
    
    -------
    rho_theta_given_y : numpy.ndarray([n_y,n_theta])
        Discrete conditional probabilities, where entry (i,j) = rho(theta_i | y_j)
    rho_xi_y : numpy.array(n_theta) 
        Discrete probability mass function for n_y values of y
    """
    rho_xi_y          = compute_total_probabilities(rho_y_given_theta , rho_theta)
    rho_theta_given_y = np.divide( rho_y_given_theta * np.tile(rho_theta,[np.shape(rho_y_given_theta)[1],1]).T , \
                                   rho_xi_y , out=np.zeros_like(rho_y_given_theta), where=rho_xi_y!=0 )
    return rho_theta_given_y , rho_xi_y

class StochasticFunction:

    def __init__(self,x,rho_x,y,cov_xy):
        """
        Takes a set of inputs (x,rho_x), set of outputs y, and a covariance matrix C(X,Y) and produces a function f(x) --> that respects those statistics.
    
        Inputs

        -------
        x : numpy.ndarray(n_x,d_x), or list
            Set of inputs x
        rho_x : numpy.ndarray(n_x)
            Discrete PDF of x
        y : numpy.ndarray(n_y,d_y), or list
            Set of possible outcomes y
        cov_xy : numpy.ndarray(n_x,n_y)
            Covariance matrix of inputs and outputs
        """
        self.x      = x
        self.rho_x  = rho_x
        self.y      = y
        self.cov_xy = cov_xy
        
    def forward(self,x_query):
        """ 
        x_query : list of x values
        y_vals : list of elements corresponding to x entries, each is a numpy.array(d_y)
        """
        idx_q         = [ self.x.index(xqj) for xqj in x_query ]
        y_vals        = []
        for i,(x_val,x_idx) in enumerate(zip(x_query,idx_q)):
            rho_y_given_x = self.cov_xy[ x_idx ]
            y_idx         = np.random.choice( np.arange(len(self.y)) , p=rho_y_given_x )
            y_vals.append( np.squeeze(self.y[y_idx]) )
        return y_vals

def compute_conditional_distribution(x,y,x_unique,y_unique):
    """
    x : list/array([n_sample,d_x])
    y : list/array([n_sample,d_y])

    rho_y_x : numpy.ndarray([n_x,n_y]) conditional probabilities of y_j given x_i
    """
    rho_y_x  = np.zeros([ len(x_unique) , len(y_unique) ])
    for (i,xi) in enumerate(x_unique):
        idx_i      = [idx    for idx in range(len(x)) if np.all(x[idx] == xi)]
        y_given_xi = [y[idx] for idx in idx_i]
        n_j        = 0
        for (j,yj) in enumerate(y_unique):
            n_ij         = len([idx for idx in range(len(y_given_xi)) if np.all(y_given_xi[idx] == yj) ])
            rho_y_x[i,j] = n_ij
            n_j         += n_ij
        rho_y_x[i] /= n_j
    return rho_y_x
