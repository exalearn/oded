import numpy as np
import matplotlib.pyplot as plt
import sdeint
from scipy.integrate import odeint
from mocu.utils.utils import *
from mocu.utils.costfunctions import *
from mocu.src.experimentaldesign import *
from mocu.src.mocu_utils import *
from mocu.scripts.visualizetoysystem import plot_points_in_full_space
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import time

class DAG( nn.Module ):
    
    def __init__( self , layers_list ):
        
        super( DAG , self ).__init__()
        
        n_layers = len( layers_list )
        
        self.f_linear = nn.ModuleList( [ nn.Linear( layers_list[i] , \
                                                    layers_list[i+1] , bias=False ) \
                                         for i in range(n_layers-1) ] )
        
    def set_weight_ij_in_layer_k( self , k , i , j , value ):
        
        f_idx = self.f_linear[ k ]
        
        f_idx.weight.data[i,j] = float(value)
        
    def multiply_weight_ij_in_layer_k( self , k , i , j , value ):
        
        f_idx = self.f_linear[ k ]
        
        f_idx.weight.data[i,j] *= float(value)
        
    def forward(self,x):
        
        for f_i in self.f_linear:
            x = f_i( x )
        
        return x
    
    def set_model_weights_with_theta( self , theta , theta_kij ):
        
        assert( len(theta_kij) == 3 )
        assert( np.size(theta) == 1 )
        
        k,i,j = theta_kij
        self.set_weight_ij_in_layer_k( k , i , j , theta )
    
    def set_model_weights_with_psi( self , psi , psi_kij ):
        
        assert( len(psi_kij) == 3 )
        assert( np.size(psi) == 1 )
        
        k,i,j = psi_kij
        self.multiply_weight_ij_in_layer_k( k , i , j , psi )
    
    def set_model_weights_with_theta_and_psi( self , psi , theta , psi_kij , theta_kij ):
    
        self.set_model_weights_with_theta( theta , theta_kij )
        self.set_model_weights_with_psi(   psi   , psi_kij )

    def get_num_layers_and_weights( self ):
        
        k = 0
        I = []
        J = []
        
        for f_i in self.f_linear:
            
            ij = f_i.weight.shape
            
            I.append( ij[0] )
            J.append( ij[1] )
            
            k += 1
            
        return k , I , J

    def set_dag_weights_from_theta( self , theta ):
        
        # Assumes theta enumeration corresponds to row-major raveling
        
        k,I,J = self.get_num_layers_and_weights( )
        
        num_weights = np.sum( [ ii * jj for (ii,jj) in zip(I,J) ] )
        
        assert( len(theta) == num_weights )
        
        count = 0
        
        for kk in range(k):
            for ii in range(I[kk]):
                for jj in range(J[kk]):
                    
                    self.set_model_weights_with_theta( theta[count] , [kk,ii,jj] )
                    
                    count += 1

    def set_dag_weights_from_psi( self , psi ):
        
        # Assumes psi enumeration corresponds to row-major raveling
        
        k,I,J = self.get_num_layers_and_weights( )
        
        num_weights = np.sum( [ ii * jj for (ii,jj) in zip(I,J) ] )
        
        assert( len(psi) == num_weights )
        
        count = 0
        
        for kk in range(k):
            for ii in range(I[kk]):
                for jj in range(J[kk]):
                    
                    self.set_model_weights_with_psi( psi[count] , [kk,ii,jj] )
                    
                    count += 1



def dag_forward( dag ):
    
    def dag_forward_round( x ):
        
        y   = dag( x ).detach().numpy()[0]
        tol = 1e-5
        
        if ( y > tol ):
            y = 1
        elif ( y < -tol ):
            y = 0
        else:
            y = np.random.choice( [0,1] )
            
        return y
    
    return dag_forward_round
            
def create_dag_cost_function( dag , n_mc , S , rho_S , psi_actions ):
    
    dag_forward_round   = dag_forward( dag )
        
    def cost( theta , psi ):
        
        assert( ( psi >= 0 ) & ( psi <= len(psi_actions) ) )
        psi = int(psi)
        
        psi_action = psi_actions[ psi ]
        
        dag.set_dag_weights_from_theta( theta )
        dag.set_dag_weights_from_psi( psi_action )
        
        J    = 0.0
        
        for i in range(n_mc):
            xi              = torch.tensor( S[ np.random.choice( np.arange(len(S)) , p=rho_S ) ] ).float()
            y_model         = dag_forward_round( xi )
            desired_state   = 0
            J              += np.abs( y_model - desired_state ) / float(n_mc)
            
        return J
    
    return cost
        


def p(I,J,n,m):
    
    assert( len(I) == len(J) )
    
    P = np.ones([n,m])
    
    for i in range( len(I) ):
        
        P[I[i],J[i]] = 0
        
    return P.ravel()

def make_random_psi( n , m , N , out ):
    
    Psi_actions = []
    
    for i in range(N):
        
        lenI = np.random.choice( np.arange( n*m ) )
        
        I = np.random.choice( np.arange(n) , lenI )
        J = np.random.choice( np.arange(m) , lenI )
        
        Psi_actions.append( np.append( p(I,J,n,m) , out ) )
    
    return Psi_actions

def make_random_psi_diagonal( n , N , out ):
    
    Psi_actions = []
    
    for i in range(N):
        
        lenI = np.random.choice( np.arange( n ) )
        
        I = np.random.choice( np.arange(n) , lenI , replace=False )
        
        Psi_actions.append( np.append( p(I,I,n,n) , out ) )
    
    return Psi_actions

    
def make_random_experiments( n_theta , n_x , p_min , p_max ):        
    
    rho_y_given_x_theta = []
    
    for i in range( n_x ):
        ri      = np.zeros( [n_theta , 2] )
        ri[:,0] = p_min
        ri[:,1] = p_max
        idx_test     = np.random.choice( np.arange(n_theta) )
        ri[idx_test] = [p_max , p_min]
        rho_y_given_x_theta.append( ri )
        
    return rho_y_given_x_theta


    
def select_experiment( n_mc=100 , n_X=10 , n_experiment=1 , n_input=10 ):
    
    np.random.seed(0)

    n_mc  = int(n_mc)
    n_psi = int(n_input) # For this problem, the two are equal
    
    layers  = [ n_input , n_input , 1 ]
    dag     = DAG( layers )
    
    t1        = np.random.choice( [-1,1] , [layers[0],layers[1]] )
    t2        = np.random.choice( [-1,1] , [layers[0],layers[1]] )
    t3        = np.random.choice( [-1,1] , [layers[0],layers[1]] )
    t4        = np.random.choice( [-1,1] , [layers[0],layers[1]] )
    
    out       = np.ones( t1.shape[0] )
    
    theta     = [ np.append(t1.ravel(),out) , np.append(t2.ravel(),out) , np.append(t3.ravel(),out) , np.append(t4.ravel(),out) ]
    rho_theta = [ 1./len(theta) for i in range(len(theta)) ]
    Theta     = dict(zip(['theta','rho_theta'] , [theta,rho_theta]))
    Psi       = np.arange( n_psi )
    
    n = layers[0]; m = layers[1]
    Psi_actions    = make_random_psi( n , m , len(Psi) , out )
    
    S         = [ np.random.choice( [0,1,-1] , layers[0] ) for i in range(4) ]
    rho_S     = [ 0.1 , 0.2 , 0.3 , 0.4 ]
    X         = np.arange(n_X)
    Y         = [ -1 , 1 ]
    E         = dict(zip(['X','Y'] , [X,Y]))
    
    rho_y_given_x_theta    = make_random_experiments( len(theta) , len(X) , 0.2 , 0.8 )
    
    theta_actual    = theta[1]
    run_real_system = lambda idx_x : np.random.choice( Y , p = rho_y_given_x_theta[idx_x][1] )
    
    print( "******** PSI ******** " )
    
    for i in range(len(Psi_actions)):
        print( Psi_actions[i] )
    
    print( "******** THETA ******** " )
    
    for i in range(len(theta)):
        print( theta[i] )

    print( "******** RHO( Y | X,THETA ) ******** " )
    
    for i in range(len(rho_y_given_x_theta)):
        print( rho_y_given_x_theta[i] )

    print('\n')
        
    mocu_data  = MOCU_data()
    
    cost       = create_dag_cost_function( dag , n_mc , S , rho_S , Psi_actions )
                        
    dt = 0

    for ii in range(n_experiment):

        print( "EXPERIMENT " + str(ii+1) + " / " + str(n_experiment) )

        t0 = time.monotonic()

        mocu_out = mocu_choose_next_experiment( Theta , Psi , E , rho_y_given_x_theta , cost )
        #mocu_out = random_choose_next_experiment( Theta , Psi , E , rho_y_given_x_theta , cost )

        dt += time.monotonic() - t0

        y_star             = run_real_system( mocu_out['idx_xstar'][0] )
        idx_outcome        = 0 if y_star <= -0.99 else 1

        mocu_data.set_data_with_current_metrics( mocu_out , idx_outcome , Theta )

        print_current_mocu_metrics( mocu_out , mocu_data.MAP_IDX[ii] , mocu_data.MAP[ii] , mocu_data.MAP_PROB[ii] )
        
    idx_min_J_design       = np.argmin( mocu_data.J_optimal )
    idx_xstar_optimal      = mocu_data.experimental_outcome[ idx_min_J_design ]
    psi_optimal            = mocu_data.psi_ibr[ idx_min_J_design ][ idx_xstar_optimal ]
    J_mocu                 = np.min( mocu_data.J_optimal )
    tx = [ ti[0] for ti in Theta['theta'] ]
    ty = [ ti[1] for ti in Theta['theta'] ]
    J_var                  = np.sum( Theta['rho_theta'] * ( ( tx - np.mean(tx) )**2 + ( ty - np.mean(ty) )**2 ) )
    J_time                 = dt
        
    return J_mocu , J_time
        
    # plt.subplot(141)
    # plt.plot( mocu_data.J_optimal ); plt.title( 'J_optimal' )

    # plt.subplot(142)
    # plt.plot( mocu_data.psi_ibr ); plt.plot( mocu_data.experimental_outcome , 'k--' )
    # plt.title( 'Psi_ibr_idx' ); plt.legend([ 'Psi(y_1)' , 'Psi(y_2)' , 'y_obs_idx' ] )

    # plt.subplot(143)
    # plt.plot( mocu_data.X_chosen); plt.title( 'X_chosen_idx' )

    # plt.subplot(144)
    # plt.plot( mocu_data.MAP_IDX );
    # plt.plot( mocu_data.MAP_PROB , 'k--' );
    # plt.title( 'MAP_theta_idx' ) ; plt.legend( [ 'MAP_idx' , 'MAP_prob' ])
    
    # plt.show()
    
    # plt.subplot(131)
    # plt.loglog( N_MC , J_design )
    # plt.xlabel( "N_MC" )
    # plt.title( 'J_design' )
    
    # plt.subplot(132)
    # plt.loglog( N_MC , J_time )
    # plt.xlabel( "N_MC" )
    # plt.title( 'J_time' )
    
    # plt.subplot(133)
    # plt.loglog( J_time , J_design )
    # plt.xlabel( "J_time" )
    # plt.ylabel( 'J_design' )
    
    # plt.show()


    
def main( ):
    
    J_mocu , J_time = select_experiment( n_input=5 )
    
    print( "J_mocu : " + str( J_mocu ) )
    print( "J_time : " + str( J_time ) + " sec" )
    
    
