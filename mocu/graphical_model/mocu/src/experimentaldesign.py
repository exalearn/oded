import numpy as np
from mocu.utils.utils import *
from scipy.interpolate import interp1d

def mocu_choose_next_experiment(Theta,Psi,E,rho_y_given_theta,cost_obj):
    """
    Selects a next experiment from discrete set of choices in E['X'] using MOCU-based sampling.
    
    Inputs
    -------
    Theta : dict['theta','rho_theta']
        Theta['theta'] : list(th_1, th_2, ..., th_{n_vals_theta}), where th_i = numpy.ndarray(n_theta) (i.e. discrete set of n_vals_theta values of theta, where theta is in R^n_theta)
        Theta['rho_theta'] : numpy.narray(n_theta) (discrete probability mass function for n_theta values of theta)
    Psi : numpy.narray(n_psi)
        Discrete set of n_psi actions
    E : dict['X','Y']
        E['X'] : list(x_1, x_2, ..., x_{n_experiments}) of experimental inputs, where x_i = numpy.ndarray(n_x)
        E['Y'] : list(y_1, y_2, ..., y_{n_outcomes}) of possible experimental outcomes
    rho_y_given_theta : list( r_1 , r_2, ..., r_n ) where r_k = numpy.ndarray([n_theta,n_y])
        List of discrete conditional probabilities, where r_k = rho(y_j | theta_i,x_k)
    cost_function : function handle to C(theta,psi)

    Outputs
    -------
    mocu_out : dict( ['x_star','avg_omega','psi_ibr'] )
        mocu_out['x_star'] : label from E['X'] of optimal experiment
        mocu_out['avg_omega'] : numpy.array( n_experiments ), average cost of each experiment
        mocu_out['psi_ibr'] : numpy.ndarray([ n_experiments , n_outcomes ]), psi_ibr[i,j] = psi_ibr( Theta | x_i|y_j )
    """

    if (type(Psi) == np.ndarray):
        assert( len(Psi.shape) == 1 )
    else:
        assert( type(Psi) == list )
        
    # Allocations

    psi_ibr   = np.zeros([ len(E['X']) , len(E['Y']) ])
    psi_ibr_idx = np.zeros([ len(E['X']) , len(E['Y']) ])
    omega     = np.zeros([ len(E['X']) , len(E['Y']) ])
    avg_omega = np.zeros(  len(E['X']) )
    rho_theta_given_x_y = []

    if (callable(cost_obj)):
        J_psi_theta = compute_cost_matrix( cost_obj , Theta , Psi )
    else:
        assert( len(cost_obj.shape) == 2 )
        J_psi_theta = cost_obj 
    
    # Main MOCU loop

    for (i,x) in enumerate(E['X']):
        
        rho_theta_given_xi_y , rho_xi_y = compute_probability_theta_given_y( rho_y_given_theta[i] , Theta['rho_theta'] )
        psi_ibr_i , psi_ibr_idx_i       = compute_psi_ibr( Psi , Theta , x , E['Y'] , rho_theta_given_xi_y , J_psi_theta )
        psi_ibr[i]                      = psi_ibr_i
        psi_ibr_idx[i]                  = psi_ibr_idx_i
        omega[i]                        = compute_conditional_avg_function( J_psi_theta , (Theta['theta'] , psi_ibr[i] , psi_ibr_idx_i) , rho_theta_given_xi_y )
        avg_omega[i]                    = np.sum( omega[i] * rho_xi_y )

        rho_theta_given_x_y.append( rho_theta_given_xi_y )

    idx_xstar  = np.argsort( avg_omega )
    X_star     = E['X'][ idx_xstar[0] ]
    mocu_out   = dict(zip( ['idx_xstar','x_star','avg_omega','psi_ibr','rho_theta_given_x_y','J_psi_theta','psi_ibr_idx'] , \
                           [idx_xstar,X_star,avg_omega,psi_ibr,rho_theta_given_x_y,J_psi_theta,psi_ibr_idx] ))
    
    return mocu_out

def compute_cost_matrix( cost_function , Theta , Psi ):
    
    J = np.zeros( [ len(Psi) , len(Theta['theta']) ] )
    
    for (i,p) in enumerate(Psi):
        
        for (j,th) in enumerate(Theta['theta']):
            
            J[i,j] = cost_function( th , p )
    
    return J

def compute_psi_ibr( psi , Theta , x , y , rho_theta_given_xi_y , cost_obj ):
    """
    Inputs

    -------
    psi : numpy.array(n_psi)
    Theta : dict['theta','rho_theta']
        Theta['theta'] : list(th_1, th_2, ..., th_{n_vals_theta}), where th_i = numpy.ndarray(n_theta) (i.e. discrete set of n_vals_theta values of theta, where theta is in R^n_theta)
        Theta['rho_theta'] : numpy.narray(n_theta) (discrete probability mass function for n_theta values of theta)
    x : numpy.ndarray(n_x) (i.e. one single experimental input)
    y : list(y_1, y_2, ..., y_{n_outcomes}) of possible experimental outcomes
    rho_theta_given_xi_y : numpy.ndarray([ n_theta , n_outcomes ])
        Discrete conditional probabilities, where entry (i,j) = rho(theta_j | y_i)
    J : np.array( [ n_psi , n_theta ] )

    Outputs

    -------
    psi_ibr : list( n_outcomes )
        psi_ibr[j] = psi_ibr( Theta | x_i,y_j )
    """

    if (callable(cost_obj)):
        J = lambda i,j,p,th : cost_obj(th,p)
    else:
        assert( len(cost_obj.shape) == 2 )
        J = lambda i,j,p,th : cost_obj[i,j]
    
    avg_cost = np.zeros([ len(psi) , np.shape(rho_theta_given_xi_y)[1] ])

    for (i,p) in enumerate(psi):

        cost_psi = np.zeros([len(Theta['theta']) , len(y)])

        for (j,th) in enumerate(Theta['theta']):
            
            J_ij = J(i,j,p,th)
            cost_psi[j] = [ J_ij for yi in y ]
            
        for (j,yj) in enumerate(y):
            avg_cost[i,j] = compute_weighted_average(cost_psi[:,j] , rho_theta_given_xi_y[:,j])

    psi_ibr_idx = np.argmin(avg_cost , axis=0)
    psi_ibr     = [ psi[idx] for idx in psi_ibr_idx ]
    return psi_ibr , psi_ibr_idx
    
def compute_conditional_avg_function( cost_obj , function_args , rho_theta_given_xi_y ):
    """
    Inputs
    
    -------
    J_psi_theta : np.array( [ n_psi , n_theta ] )
    function_args : tuple( theta , x , psi_ibr ) -- these are inputs to 'cost_function'
        theta   : list(th_1, th_2, ..., th_{n_vals_theta}), where th_i = numpy.ndarray(n_theta) (i.e. discrete set of n_vals_theta values of theta, where theta is in R^n_theta)
        x       : numpy.ndarray(n_x) (i.e. one single experimental input)
        psi_ibr : numpy.array(n_outcomes) (i.e. psi_ibr for experiment x)
    rho_theta_given_xi_y : numpy.ndarray([ n_theta , n_outcome ]) , rho(theta_i | x,y_j )

    Outputs
    
    -------
    conditional_avg_func : numpy.array(n_outcomes)
        Conditional average E_{theta|y}[ C(theta,psi_ibr) ]
    """
    
    if (callable(cost_obj)):
        J = lambda i,j,p,th : cost_obj(th,p)
    else:
        assert( len(cost_obj.shape) == 2 )
        J = lambda i,j,p,th : cost_obj.T[i,j] # Transpose needed since here, cost_obj = J(psi_i , theta_j)
    
    theta, psi_ibr , psi_ibr_idx = function_args
    
    ny                   = np.shape(rho_theta_given_xi_y)[1]
    conditional_costs    = np.zeros([ len(theta) , len(psi_ibr) ])
    conditional_avg_func = np.zeros( len(psi_ibr) )
    
    for (i,th) in enumerate(theta):
        
        for (j,p) in enumerate(psi_ibr):
            conditional_costs[i,j] = J( i , psi_ibr_idx[j] , p , th )
            #conditional_costs[i,j] = J_psi_theta[ p , th ]
            #conditional_costs[i,j] = cost_function( th , p )

    for j in range(len(psi_ibr)):
        conditional_avg_func[j] = compute_weighted_average( conditional_costs[:,j] , rho_theta_given_xi_y[:,j] )
    
    return conditional_avg_func

def output_mocu_results_by_cost( mocu_out , E ):
    
    idx_sort = np.argsort( mocu_out['avg_omega'] )

    print('******** CANDIDATE EXPERIMENTS ********')

    for (i,idx) in enumerate(idx_sort):

        print( 'X_' + str(idx+1) + ' : ' + str(E['X'][idx]) )
        print( 'avg_omega = ' + str(mocu_out['avg_omega'][idx]) )
        print( 'psi_ibr = ' + str(mocu_out['psi_ibr'][idx]) )
        print( 'rho_theta_given_x_y = \n' + str(mocu_out['rho_theta_given_x_y'][idx]) )

    print('***************************************\n')



def random_choose_next_experiment(Theta,Psi,E,rho_y_given_theta,cost_obj):
    """
    Selects a next experiment from discrete set of choices in E['X'] using random sampling.
    
    Inputs
    -------
    Theta : dict['theta','rho_theta']
        Theta['theta'] : list(th_1, th_2, ..., th_{n_vals_theta}), where th_i = numpy.ndarray(n_theta) (i.e. discrete set of n_vals_theta values of theta, where theta is in R^n_theta)
        Theta['rho_theta'] : numpy.narray(n_theta) (discrete probability mass function for n_theta values of theta)
    Psi : numpy.narray(n_psi)
        Discrete set of n_psi actions
    E : dict['X','Y']
        E['X'] : list(x_1, x_2, ..., x_{n_experiments}) of experimental inputs, where x_i = numpy.ndarray(n_x)
        E['Y'] : list(y_1, y_2, ..., y_{n_outcomes}) of possible experimental outcomes
    rho_y_given_theta : list( r_1 , r_2, ..., r_n ) where r_k = numpy.ndarray([n_theta,n_y])
        List of discrete conditional probabilities, where r_k = rho(y_j | theta_i,x_k)
    cost_function : function handle to C(theta,psi)

    Outputs
    -------
    mocu_out : dict( ['x_star','avg_omega','psi_ibr'] )
        mocu_out['x_star'] : label from E['X'] of optimal experiment
        mocu_out['avg_omega'] : numpy.array( n_experiments ), average cost of each experiment
        mocu_out['psi_ibr'] : numpy.ndarray([ n_experiments , n_outcomes ]), psi_ibr[i,j] = psi_ibr( Theta | x_i|y_j )
    """
    
    if (type(Psi) == np.ndarray):
        assert( len(Psi.shape) == 1 )
    else:
        assert( type(Psi) == list )
    
    # Allocations
    
    psi_ibr   = np.zeros([ len(E['X']) , len(E['Y']) ])
    psi_ibr_idx = np.zeros([ len(E['X']) , len(E['Y']) ])
    omega     = np.zeros([ len(E['X']) , len(E['Y']) ])
    avg_omega = np.zeros(  len(E['X']) )
    rho_theta_given_x_y = []

    if (callable(cost_obj)):
        J_psi_theta = compute_cost_matrix( cost_obj , Theta , Psi )
    else:
        assert( len(cost_obj.shape) == 2 )
        J_psi_theta = cost_obj 
    
    # Main sampling loop
    
    for (i,x) in enumerate(E['X']):
        
        rho_theta_given_xi_y , rho_xi_y = compute_probability_theta_given_y( rho_y_given_theta[i] , Theta['rho_theta'] )
        psi_ibr_idx_i                   = np.random.choice( np.arange( len(Psi) ) , len(E['Y']) )
        psi_ibr[i]                      = Psi[psi_ibr_idx_i] # Select a random psi
        psi_ibr_idx[i]                  = psi_ibr_idx_i
        omega[i]                        = compute_conditional_avg_function( J_psi_theta , (Theta['theta'] , psi_ibr[i] , psi_ibr_idx_i) , rho_theta_given_xi_y )
        avg_omega[i]                    = np.sum( omega[i] * rho_xi_y )
        
        rho_theta_given_x_y.append( rho_theta_given_xi_y )
        
    idx_xstar  = np.argsort( avg_omega )
    np.random.shuffle( idx_xstar ) # Select a random experiment, not necessarily optimal one
    X_star     = E['X'][ idx_xstar[0] ]
    mocu_out   = dict(zip( ['idx_xstar','x_star','avg_omega','psi_ibr','rho_theta_given_x_y','J_psi_theta','psi_ibr_idx'] , \
                           [idx_xstar,X_star,avg_omega,psi_ibr,rho_theta_given_x_y,J_psi_theta,psi_ibr_idx] ))

    
    return mocu_out
