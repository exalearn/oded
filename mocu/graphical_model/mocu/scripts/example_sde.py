import numpy as np
import matplotlib.pyplot as plt
import sdeint
from scipy.integrate import odeint
from mocu.utils.utils import *
from mocu.utils.costfunctions import *
from mocu.src.experimentaldesign import *
from mocu.scripts.visualizetoysystem import plot_points_in_full_space

def make_rhs_full_system(a,b,k,c,lam,psi,theta):
    
    def rhs_full_system(y,t):
        C      = c(a,b,k,y[0],psi,theta)
        y1_dot = lam[0] * (y[0] - 1)
        y2_dot = lam[1] * (y[1] - C) * (y[1] - a) * (y[1] - b)
        return [y1_dot , y2_dot]

    return rhs_full_system

def make_f(a,b,c):
    def f(y,t):
        return -(y-c) * (y-a) * (y-b)
    return f

def make_noise_function(theta):
    def g(y,t):
        return theta
    return g

def make_stochastic_model( theta , psi ):

    theta_min = 0.; theta_max = 1.0
    psi_min   = 0.; psi_max   = 1.0
    theta_01  = ( theta - theta_min ) / ( theta_max - theta_min )
    psi_01    = ( psi   - psi_min   ) / ( psi_max   - psi_min )
    delta_pt  = np.abs( psi_01 - theta_01 )
    root      = 0.04 * delta_pt + 0.48 # \in [0.48,0.52]
    
    f_model     = make_f( 0 , 1 , root )
    g_model     = make_noise_function( 0.03 )
    
    return f_model,g_model

def run_model_system( psi , theta , x ):

    x = x[1]
    
    dt      = 0.05
    tf      = 10
    t       = np.arange(0,tf,dt)
    f_model,g_model = make_stochastic_model( theta , psi )
    y_model = np.squeeze( sdeint.itoint( f_model , g_model , x , t ) )
    y_threshold  = int(np.round(y_model[-1]))
    
    return y_threshold

def run_deterministic_system( psi , theta , x ):
    
    dt      = 0.05
    tf      = 10
    t       = np.arange(0,tf,dt)
    lam     = [-0.01,-1]
    k       = 5
    a = 0; b=1;
    
    c      = lambda a,b,k,y1,psi,theta : (0.48 + 0.04*np.abs(psi-theta) ) + 0.04*np.abs(b-a)*np.sin(2*np.pi*k*y1)
    f_real = make_rhs_full_system(a,b,k,c,lam,psi,theta)
    
    y_model = np.squeeze( odeint( f_real , x , t ) )
    y_real  = int(np.round( y_model[-1][1] ))
    
    return y_real

def run_real_system_deterministic( psi , x ):
    
    theta_true = 0.5
    
    return run_deterministic_system( psi , theta_true , x )    


def run_real_system_stochastic( psi , x ):
    
    theta_true = 0.5
    
    return run_model_system( psi , theta_true , x )    

def cost( theta , x , psi):
    # Computes C( theta,psi ; x )
    # Does this by monte carlo sampling over a uniform measure

    n_mc = 100
    J    = 0.0
    
    for i in range(n_mc):
        y_model         = run_model_system( psi , theta , x )
        desired_state   = 1
        J              += np.abs( y_model - desired_state ) / float(n_mc)

    return J
    

def main():
    
    # Prior knowledge: discrete ranges/distributions for (theta,psi)
    # Prior knowledge: set of experiments X with possible outcomes Y
    # In this example, we do not know rho_j(y_j|(theta_i,x_k)) and must approximate it with MC sampling

    theta     = np.array([ 0.0 , 0.5 , 1.0])
    rho_theta = 1./len(theta) * np.ones_like(theta)
    Theta     = dict(zip(['theta','rho_theta'] , [theta,rho_theta]))
    Psi       = [0.,0.5,1.0]
    n_experiment       = 100
    n_sample_theta     = 100
    X_chosen           = []
    J_optimal          = []
    psi_ibr            = []
    experimental_outcome = []
    MAP                  = []
    
    dt     = 0.05
    tf     = 10
    t      = np.arange(0,tf,dt)
    
    psi_0 = 0.0
    
    plt.figure(1); plt.ion(); plt.show()    
    for ii in range(n_experiment):
        
        # Draw from experiment space
        print( 'Drawing MC samples from candidate space of experiments...' )
        
        X                      = tuple(zip( np.random.uniform(0,1,3) , \
                                            np.array([0.48,0.5,0.52]) ))
        Y                      = [0,1]
        E                      = dict(zip(['X','Y'] , [X,Y]))
        
        # Plot current stuff
        plt.clf()
        if (ii != 0):
            plt.figure(1); plt.subplot(131);
            plt.plot( np.arange(1,len(MAP)+1) , [mapi for mapi in MAP] , 'b' )
            plt.plot( np.arange(1,len(MAP)+1) , [xi[1] for xi in X_chosen] , 'r' )
            plt.legend( ['MAP' + r'$(\theta)$' , r'$x_0$'] )
            plt.xlabel('iteration',fontsize=16)
            plt.subplot(132)
            plt.plot( np.arange(1,len(psi_ibr)+1) , [pi[0] for pi in psi_ibr] , 'b' )
            plt.plot( np.arange(1,len(psi_ibr)+1) , [pi[1] for pi in psi_ibr] , 'r' )
            plt.plot( np.arange(1,len(psi_ibr)+1) , [yi    for yi in experimental_outcome] , 'k--' )
            plt.xlabel('iteration',fontsize=16)
            plt.legend( [r'$\psi(\Theta | y=0)$',r'$\psi(\Theta | y=1)$','y'] )

        print( Theta['theta'] )
        print( Theta['rho_theta'] )
        plt.figure(1); plt.subplot(133); plt.bar( Theta['theta'] , Theta['rho_theta'] , width = 0.5*( Theta['theta'][1]-Theta['theta'][0] ) )
        plt.xlabel(r'$\theta$',fontsize=16)
        plt.ylabel(r'$\rho(\theta | (x,y))$',fontsize=16)
        plt.draw(); plt.pause(0.001)

        # EXPERIMENT SPACE
        # Approximate rho(y | theta , x_i , psi_0) for each x_i using fixed psi_0
        rho_y_given_theta  = []
        theta_samples      = []
        samples            = []
        for (i,x) in enumerate(E['X']):
            y_sample     = []
            theta_sample = []
            y_samples_over_theta = []
            for (j,th) in enumerate(Theta['theta']):
                y_sample_k = []
                for k in range(n_sample_theta):
                    y_threshold = run_model_system( psi_0 , th , x )
                    y_sample_k.append(   y_threshold )
                    y_sample.append(     y_threshold )
                    theta_sample.append( th          )
                y_samples_over_theta.append( y_sample_k )
            samples.append( y_samples_over_theta )
            print('computing rho(y | theta , X_' + str(i+1) + ')')
            rho_y_given_theta.append( compute_conditional_distribution( theta_sample , \
                                                                        y_sample , \
                                                                        Theta['theta'] , \
                                                                        E['Y'] ) )
            
        # Compute next experiment with mocu
        mocu_out = mocu_choose_next_experiment(Theta, Psi, E, rho_y_given_theta, cost)
        
        # Choose optimal experiment and update posterior rho_theta_given_x_y
        print( "CHOSEN EXPERIMENT: X_" + str(mocu_out['idx_xstar'][0]+1) + ' : ' + str(mocu_out['x_star']) )
        print( 'avg_omega_xstar = ' + str(mocu_out['avg_omega'][mocu_out['idx_xstar'][0]]) )
        print( 'psi_ibr_xstar = ' + str(mocu_out['psi_ibr'][mocu_out['idx_xstar'][0]] ) )
        print( 'rho_theta_given_xstar_y = \n' + str(mocu_out['rho_theta_given_x_y'][mocu_out['idx_xstar'][0]]  ) )

        # Run the chosen experiment and select psi_ibr( x_star , y_star )
        y_star       = run_real_system_deterministic( psi_0 , mocu_out['x_star'] )
        idx_outcome  = 0 if y_star < 0.5 else 1
        psi_ibr_star = mocu_out['psi_ibr'][mocu_out['idx_xstar'][0]][idx_outcome]
        Theta['rho_theta'] = mocu_out['rho_theta_given_x_y'][mocu_out['idx_xstar'][0]][:,idx_outcome]
        mapi               = Theta['theta'][np.argmax(Theta['rho_theta'])]
        print(mapi)
        
        # Save chosen experiment and use it to update posterior on theta
        X_chosen.append( mocu_out['x_star'] )
        J_optimal.append( mocu_out['avg_omega'][mocu_out['idx_xstar'][0]] )
        psi_ibr.append( mocu_out['psi_ibr'][mocu_out['idx_xstar'][0]] )
        experimental_outcome.append( y_star )
        MAP.append( mapi )
        
    plt.ioff(); plt.show()
    
    # # Output results
    # output_mocu_results_by_cost(mocu_out, E)
    # plot_points_in_full_space([ [x[0] for x in E['X']] , [x[1] for x in E['X']]  ] , c , colors=mocu_out['avg_omega'])
    # plt.show()
    
    
if __name__ == '__main__':
    main()
