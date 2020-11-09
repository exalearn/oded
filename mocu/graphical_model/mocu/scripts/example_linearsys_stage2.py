import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mocu.utils.utils import *
from mocu.utils.costfunctions import *
from mocu.src.experimentaldesign import *
from mocu.src.mocu_utils import *
from mocu.scripts.visualizetoysystem import plot_points_in_full_space
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
import time
import argparse
    
def make_random_experiments( n_theta , n_x , p_min , p_max ):
    
    assert( n_x <= n_theta )
    
    idx_tests = np.random.choice( np.arange(n_theta) , n_x , replace=False )
    
    rho_y_given_x_theta = []
    
    for i in range( n_x ):
        
        ri      = np.zeros( [n_theta , 2] )
        ri[:,0] = p_min
        ri[:,1] = p_max
        idx_test     = idx_tests[i]
        ri[idx_test] = [p_max , p_min]
        rho_y_given_x_theta.append( ri )
        
    return rho_y_given_x_theta

def generate_random_eigenvalue_distributions( n_theta , n_s ):
    
    log_L_mean = lambda a,b : 1 - ( 1 - (1e-2)*np.random.uniform(1,100) )*np.heaviside( np.arange(n_s) - np.random.uniform(a,b)*n_s , 0.5 )
    l_rand     = lambda mu,sig : [ li * ( 1 + np.random.normal(0,sig) ) for li in mu ]
    theta      = [ np.diag( l_rand( log_L_mean(0.2,0.8) , 0.1 ) ) for i in range(n_theta) ]
    
    return theta

def create_linsys_cost_function( S , psi_actions ):
    
    assert( len(S.shape) == 2 )
    
    n_s     = np.shape(S)[0]
    cost    = lambda A,x : np.mean( np.linalg.norm( A.dot(S) , axis=0 )**2 / np.linalg.norm( S , axis=0 )**2 )
    
    V       = ortho_group.rvs( n_s )
    
    def cost_func( theta , psi ):
        
        assert( ( psi >= 0 ) & ( psi <= len(psi_actions) ) )
        assert( theta.shape == V.shape )
        
        psi        = int(psi)
        psi_action = psi_actions[ psi ]
        
        A_theta = V.dot( theta ).dot( V.T )
        A       = psi_action.T.dot( A_theta ).dot( psi_action )
        J       = cost( A , S )
        
        return J
    
    return cost_func
    

def plot_mocu_output( mocu_data , th_actual_idx ):
    
    plt.subplot(231)
    plt.plot( mocu_data.J_optimal ); plt.title( 'J_optimal' )
    J_actual = [ mocu_data.J_psi_theta[ int(pi) , th_actual_idx ] for pi in mocu_data.psi_ibr_idx ]
    plt.plot( J_actual )
    plt.legend([ 'J_theta' , 'J_actual' ])
        
    plt.subplot(232)
    plt.plot( mocu_data.psi_ibr );
    plt.plot( mocu_data.psi_ibr_idx , 'r--' )
    #plt.plot( mocu_data.experimental_outcome , 'k--' )
    plt.title( 'Psi_ibr_idx' ); plt.legend([ 'Psi(y_1)' , 'Psi(y_2)' , 'Psi_xstar' ] )
    
    plt.subplot(233)
    plt.plot( mocu_data.X_chosen); plt.title( 'X_chosen_idx' )
    
    plt.subplot(234)
    plt.plot( mocu_data.th_mean )
    plt.plot( mocu_data.MAP_IDX );
    plt.fill_between( np.arange( len(mocu_data.MAP_IDX) ) , \
                      np.array( mocu_data.th_mean ) - np.array( [ ti[0] for ti in mocu_data.t_68 ] ) , \
                      np.array( mocu_data.th_mean ) + np.array( [ ti[1] for ti in mocu_data.t_68 ] ) , \
                      color='b' , \
                      alpha = 0.4 )
    plt.fill_between( np.arange( len(mocu_data.MAP_IDX) ) , \
                      np.array( mocu_data.th_mean ) - np.array( [ ti[0] for ti in mocu_data.t_95 ] ) , \
                      np.array( mocu_data.th_mean ) + np.array( [ ti[1] for ti in mocu_data.t_95 ] ) , \
                      color='b' , \
                      alpha = 0.2 )
    plt.gca().set_ylim( [ 0 , len(mocu_data.theta) ] )
    plt.legend( [ 'mean_idx' , 'MAP_idx' ] )
    
    plt.subplot(235)
    plt.plot( mocu_data.MAP_PROB );
    plt.title( 'MAP_prob' )
    
    plt.subplot(236)
    plt.bar( np.arange( len(mocu_data.rho_theta[-1]) ) , mocu_data.rho_theta[-1] )
    plt.title( 'Final rho_theta' )
    
    plt.show()

    

def select_experiment( n_s=16 , n_experiment=16 , n_mc=128 , n_theta=16 , n_psi=16 , method='mocu' ):
    
    if method == 'mocu':
        selection_func = mocu_choose_next_experiment
    else:
        selection_func = random_choose_next_experiment
    
    n_X          = n_theta
    
    theta     = generate_random_eigenvalue_distributions( n_theta , n_s )
    rho_theta = [ 1./len(theta) for i in range(len(theta)) ]
    Theta     = dict(zip(['theta','rho_theta'] , [theta,rho_theta]))
    
    Psi         = np.arange( n_psi )
    Psi_actions = [ ortho_group.rvs( n_s ) for i in range(n_psi) ]
    
    S         = np.random.uniform( 0 , 1 , [n_s,n_mc] )
    
    X         = np.arange(n_X)
    Y         = [ -1 , 1 ]
    E         = dict(zip(['X','Y'] , [X,Y]))
    
    rho_y_given_x_theta = make_random_experiments( len(theta) , len(X) , 0.1 , 0.9 )

    th_actual_idx   = n_theta//2
    theta_actual    = theta[th_actual_idx]
    run_real_system = lambda idx_x : np.random.choice( Y , p = rho_y_given_x_theta[idx_x][th_actual_idx] )
    
    mocu_data  = MOCU_data()
    
    cost        = create_linsys_cost_function( S , Psi_actions )
    J_psi_theta = compute_cost_matrix( cost , Theta , Psi )

    J_actual_optimal = np.min( J_psi_theta[:,th_actual_idx] ) 
    
    dt = 0
    
    for ii in range(n_experiment):
        
        # print( "EXPERIMENT " + str(ii+1) + " / " + str(n_experiment) )
        
        t0 = time.monotonic()
        
        mocu_out = selection_func( Theta , Psi , E , rho_y_given_x_theta , J_psi_theta )
        
        dt += time.monotonic() - t0
        
        y_star             = run_real_system( mocu_out['idx_xstar'][0] )
        idx_outcome        = 0 if y_star <= -0.99 else 1
        
        mocu_data.set_data_with_current_metrics( mocu_out , idx_outcome , Theta )
        
        #print_current_mocu_metrics( mocu_out , mocu_data.MAP_IDX[ii] , mocu_data.MAP[ii] , mocu_data.MAP_PROB[ii] )
                
    idx_min_J_design       = np.argmin( mocu_data.J_optimal )
    idx_xstar_optimal      = mocu_data.experimental_outcome[ idx_min_J_design ]
    psi_optimal            = mocu_data.psi_ibr[ idx_min_J_design ][ idx_xstar_optimal ]
    J_mocu                 = np.min( mocu_data.J_optimal )
    J_time                 = dt
    J_actual               = [ ( mocu_data.J_psi_theta[ int(pi) , th_actual_idx ] - J_actual_optimal ) / J_actual_optimal for pi in mocu_data.psi_ibr_idx ]
    
    #plot_mocu_output( mocu_data , th_actual_idx )
    
    return J_actual , psi_optimal , mocu_out['J_psi_theta']
        


def test_random_similarity_transforms( n_theta=32 , n_psi=32 ):
    
    n_s     = 16
    n_mc    = 128
    
    #l_mean  = [ 10**(-2*i/n_s) for i in range(n_s) ]
    #l_rand     = lambda mu,sig : [ li * ( 1 + np.random.normal(0,sig) ) for li in mu ]
    #theta      = [ np.diag( l_rand(l_mean,0.1) ) for i in range(n_theta) ]
    log_L_mean = lambda a,b : 1 - ( 1 - (1e-2)*np.random.uniform(1,100) )*np.heaviside( np.arange(n_s) - np.random.uniform(a,b)*n_s , 0.5 )
    l_rand     = lambda mu,sig : [ li * ( 1 + np.random.normal(0,sig) ) for li in mu ]
    theta      = [ np.diag( l_rand( log_L_mean(0.2,0.8) , 0.1 ) ) for i in range(n_theta) ]
    
    V       = ortho_group.rvs( n_s )
    
    A_theta = [ V.dot( ti ).dot( V.T ) for ti in theta ]
    
    Psi     = [ ortho_group.rvs( n_s ) for i in range(n_psi) ]
    
    s       = np.random.uniform( 0 , 1 , [n_s,n_mc] )
    
    cost    = lambda A,x : np.mean( np.linalg.norm( A.dot(x) , axis=0 )**2 / np.linalg.norm( x , axis=0 )**2 )
    
    J = []
    for i,ti in enumerate(theta):
        
        J_ti = []
        
        for j,pj in enumerate(Psi):
            
            A_ij = pj.T.dot( A_theta[i] ).dot( pj )
            J_ti.append( cost( A_ij , s ) )
            
        J.append( J_ti )
    
    return J , theta , Psi , V


def compute_optimality_criterion( psi , V , x_mean ):
    
    y = V.T.dot( psi ).dot( x_mean )
    
    return y
    
    
def run_test( n_theta , n_psi ):
    
    np.random.seed(0)
    
    J_theta , theta , psi , V = test_random_similarity_transforms( n_theta , n_psi )
    J         = np.ravel(J_theta)
    J_psi     = np.array( J_theta ).T
    
    meant = np.ravel( [ np.mean( np.diag(ti**2) ) * np.ones(n_psi) for ti in theta ] )
    maxt  = np.ravel( [ np.max( np.diag(ti**2) ) * np.ones(n_psi) for ti in theta ] )
    mint  = np.ravel( [ np.min( np.diag(ti**2) ) * np.ones(n_psi) for ti in theta ] )
    idx   = np.argsort(meant)
    EJ_T  = np.mean( J_psi , axis=1 )
    idx_opt = np.argsort( EJ_T )
    J_opt = J_psi[idx_opt[0]]

    print( EJ_T[idx_opt[0]] , idx_opt[0] )
    
    meant_tmp = np.ravel( [ np.mean( np.diag(ti**2) ) for ti in theta ] )
    idx_tmp   = np.argsort( meant_tmp )
    
    plt.subplot(231); plt.hist( J , 20 ); plt.ylabel( 'J(T,P)' )
    plt.subplot(232); plt.hist( meant , 20 ); plt.ylabel( 'mean(L^2)' )
    plt.subplot(233);
    plt.plot( meant , J , 'ko' )
    plt.plot( meant[idx] , meant[idx] , 'b' );
    plt.plot( meant_tmp[idx_tmp] , J_opt[idx_tmp] , 'r.-' )
    plt.fill_between( meant[idx] , mint[idx] , maxt[idx] , alpha=0.1 , color='b' )
    plt.xlabel( 'mean(L^2)' ); plt.ylabel( 'J(T,P)' )
    
    plt.subplot(234)
    for i,ji in enumerate(J_theta):
        plt.hist( ji ,  alpha=0.3 )#, color='b' )
    
    plt.ylabel( 'J(T)' )
    
    plt.subplot(235)
    plt.hist( EJ_T[idx_opt] , 20 )
    plt.ylabel( 'sort( E_T[ J ] )' )
    
    plt.subplot(236)
    for i,ti in enumerate(theta):
        plt.semilogy( np.diag(ti**2) )
    
    plt.ylabel( 'L_i^2' )
    
    n_s    = np.shape( psi[idx_opt[0]] )[0]
    x_mean = np.ones( n_s ) / np.sqrt( n_s )
    y_opt  = compute_optimality_criterion( psi[idx_opt[0]] , V , x_mean )
    
    plt.figure()
    for i,ti in enumerate(theta):
        plt.scatter( np.diag(ti**2) , y_opt**2 )
    
    # plt.figure()
    # plt.arrow( 0,0 , V[0,0] , V[1,0] , color='r' )
    # plt.arrow( 0,0 , V[0,1] , V[1,1] , color='b' )
    # plt.arrow( 0,0 , psi[idx_opt[0]][0,0] , psi[idx_opt[0]][1,0] , color='r' , ls='--' )
    # plt.arrow( 0,0 , psi[idx_opt[0]][0,1] , psi[idx_opt[0]][1,1] , color='b' , ls='--' )
    # plt.gca().set_aspect('equal')
    
    plt.show()


def compute_oed_error_stats( n_s , n_experiment , n_mc , n_theta , n_psi , n_oed , oed_type ):
    
    avg_error    = []
    avg_var      = []
    
    J_avg        = np.zeros( n_experiment )
    J_all        = []
    
    for i in range(n_oed):
        
        print( "OED RUN " + str(i+1) + " / " + str(n_oed) )
        
        J_oed , psi_optimal , J_psi_theta = select_experiment( n_s , n_experiment , n_mc , n_theta , n_psi , oed_type )
        Ji     = np.array( J_oed )
        J_avg += Ji
        J_all.append( Ji )
    
    J_avg /= n_oed
    J_all  = np.array( J_all )
    
    J_68 = np.zeros( [ n_experiment , 2 ] )
    J_95 = np.zeros( [ n_experiment , 2 ] )
    J_50 = np.zeros( n_experiment )
    
    for j,Jj in enumerate(J_all.T):
       
        #print( "Wenyi29" )
 
        t_68 = [ np.percentile( Jj , 16 )  , np.percentile( Jj , 16+68 ) ]
        t_95 = [ np.percentile( Jj , 2.5 ) , np.percentile( Jj , 2.5+95 ) ]
        
        J_68[j]  = t_68
        J_95[j]  = t_95
        J_50[j]  = np.percentile( Jj , 50 )
            
    return J_avg , J_68 , J_95 , J_50 , J_all
    
    
def main( n_theta , n_psi , n_mc , n_s , n_experiment , n_oed , oed_type ):
        
    #np.random.seed(0)
    
    #J_oed , psi_optimal , J_psi_theta = select_experiment( n_s , n_experiment , n_mc , n_theta , n_psi )
    #plt.imshow( J_psi_theta.T )
    
    #run_test( n_theta , n_psi )
    
    ## Unrolling function call compute_oed_error_stats() here.

    avg_error = []
    avg_var   = []
    J_avg     = np.zeros(n_experiment)
    J_all     = []

    '''print("OED RUN " + str(i+1) + " / " + str(n_oed))
    J_oed, psi_optimal, J_psi_theta = select_experiment(n_s, n_experiment, n_mc, n_theta, n_psi, oed_type)
    Ji = np.array(J_oed)
    np.save('Ji_{}.npy'.format(i+1), Ji)'''

    for i in range(n_oed):
        Ji = np.load('Ji_{}.npy'.format(i+1))
        J_avg += Ji
        J_all.append(Ji)

    J_avg /= n_oed
    J_all = np.array(J_all)
    J_68 = np.zeros([n_experiment, 2])
    J_95 = np.zeros([n_experiment, 2])
    J_50 = np.zeros(n_experiment)

    for j, Jj in enumerate(J_all.T):
       t_68 = [np.percentile(Jj, 16), np.percentile(Jj, 16+68)]
       t_95 = [np.percentile(Jj, 2.5), np.percentile(Jj, 2.5+95)]
       J_68[j] = t_68
       J_95[j] = t_95
       J_50[j] = np.percentile(Jj, 50)

    ##J_mean , J_68 , J_95 , J_50 , J_all = compute_oed_error_stats( n_s , n_experiment , n_mc , n_theta , n_psi , n_oed , oed_type )
    # J_50 = np.load( 'J_median.npy' )
    # J_68 = np.load( 'J_68.npy' )
    # J_95 = np.load( 'J_95.npy' )
    
    plt.figure( figsize=(1.414*5,5) )
    
    plt.semilogy( J_50 )
    plt.fill_between( np.arange( n_experiment ) , J_68[:,0] , J_68[:,1] , alpha=0.2 , color='b' )
    plt.fill_between( np.arange( n_experiment ) , J_95[:,0] , J_95[:,1] , alpha=0.1 , color='b' )
    plt.ylim( [ 1e-3 , 2e0 ] )
    plt.xlabel( r'$i$' , fontsize=20 )
    plt.ylabel( r'J($\theta^* , \psi^{(i)}}$)' , fontsize=20 )
    plt.tight_layout()
    
    plt.savefig( 'oed.png' )

    np.save( 'J_median.npy' , J_50 )
    np.save( 'J_68.npy'     , J_68 )
    np.save( 'J_95.npy'     , J_95 )
    
    #plt.show()
    
if __name__ == '__main__':
    
    '''
    n_theta      = 16
    n_psi        = 16
    n_mc         = 128
    n_s          = 16
    n_experiment = 32
    n_oed        = 128
    oed_type     = 'mocu'
    '''
    
    parser = argparse.ArgumentParser(description="MOCU")
    parser.add_argument("--num_run", "-n", help="number of OED runs")
    parser.add_argument("--theta", "-t", help="value of theta")
    parser.add_argument("--psi", "-p", help="value of Psi")
    parser.add_argument("--s", "-s", help="value of s")
    args = parser.parse_args()

    if args.num_run is None:
        sys.exit("\nNumber of OED runs is not provided.\n")
    else:
        n_oed = int(args.num_run)

    if args.theta is None:
        sys.exit("\nValue of Theta is not provided.\n")
    else:
        n_theta = int(args.theta)

    if args.psi is None:
        sys.exit("\nValue of Psi is not provided.\n")
    else:
        n_psi = int(args.psi)

    if args.s is None:
        sys.exit("\nValue of S is not provided.\n")
    else:
        n_s = int(args.s)

    n_mc         = 128
    n_experiment = 32
    oed_type     = 'mocu'

    main( n_theta , n_psi , n_mc , n_s , n_experiment , n_oed , oed_type )
