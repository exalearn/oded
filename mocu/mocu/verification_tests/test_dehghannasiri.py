import numpy as np
import matplotlib.pyplot as plt
from mocu.utils.utils import *
from mocu.utils.costfunctions import *
from mocu.src.experimentaldesign import *
from mocu.src.mocu_utils import *
from mocu.scripts.visualizetoysystem import plot_points_in_full_space
import torch
import torch.nn as nn

# ***********************************************************
# 
# VERIFICATION TEST TO APPROXIMATE RESULTS FROM:
# "Optimal Experimental Design for Materials Discovery",
# Dehghannasiri et al., 2017
#
# ***********************************************************


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
        






def test_dehghannasiri_2state_DAG( ):
        
    layers  = [ 2 , 1 ]
    dag     = DAG( layers )
    
    theta     = [ (-1,-1) , (-1,1) , (1,-1) , (1,1) ]
    rho_theta = [ 0.35 , 0.3 , 0.25 , 0.1 ]
    Theta     = dict(zip(['theta','rho_theta'] , [theta,rho_theta]))
    Psi       = [ 0 , 1 ]
    Psi_actions = [ (1,0) , (0,1) ]
    S         = [ (0,0) , (0,1) , (1,0) , (1,1) ]
    rho_S     = [ 0.1 , 0.2 , 0.3 , 0.4 ]
    X         = [ 0 , 1 ]
    Y         = [ -1 , 1 ]
    E         = dict(zip(['X','Y'] , [X,Y]))
    n_mc      = 1000
    
    cost      = create_dag_cost_function( dag , n_mc , S , rho_S , Psi_actions )
    
    rho_y_given_x1      = np.array( [ [0.6,0.4] , [0.6,0.4] , [0.2,0.8] , [0.2,0.8] ] )
    rho_y_given_x2      = np.array( [ [0.8,0.2] , [0.4,0.6] , [0.8,0.2] , [0.4,0.6] ] )
    rho_y_given_x_theta = [ rho_y_given_x1 , rho_y_given_x2 ]
    
    theta_actual    = theta[1]
    run_real_system = lambda idx_x : np.random.choice( Y , p = rho_y_given_x_theta[idx_x][1] )
    
    mocu_data       = MOCU_data()
            
    mocu_out = mocu_choose_next_experiment(Theta, Psi, E, rho_y_given_x_theta, cost)
        
    y_star             = run_real_system( mocu_out['idx_xstar'][0] )
    idx_outcome        = 0 if y_star <= -0.99 else 1
        
    mocu_data.set_data_with_current_metrics( mocu_out , idx_outcome , Theta )
        
    return mocu_out , mocu_data




def main():
    
    mocu_out , mocu_data = test_dehghannasiri_2state_DAG()
    
    print_current_mocu_metrics( mocu_out , mocu_data.MAP_IDX[0] , mocu_data.MAP[0] , mocu_data.MAP_PROB[0] )
    
    tol = 1e-2
    
    avg_omega_1 = 0.3460
    avg_omega_2 = 0.3550
    
    if ( abs( mocu_out['avg_omega'][0] - avg_omega_1 ) < tol ):
        print( 'avg_omega for optimal experiment is correct to accuracy ' + str(tol) )
    else:
        print( 'ERROR: avg_omega for optimal experiment is not correct, should be: ' + str( avg_omega_1 ) )
        
    if ( abs( mocu_out['avg_omega'][1] - avg_omega_2 ) < tol ):
        print( 'avg_omega for sub-optimal experiment is correct to accuracy ' + str(tol) )
    else:
        print( 'ERROR: avg_omega for sub-optimal experiment is not correct, should be: ' + str( avg_omega_2 ) )

    print( '\n' )

    
if __name__ == '__main__':

    main()    
