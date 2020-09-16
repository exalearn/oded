import unittest
import numpy as np
import sys,os
import matplotlib.pyplot as plt
from mocu.utils.utils import *
from mocu.utils.costfunctions import *
from mocu.src.experimentaldesign import *
import warnings

class MocuTestMultivariate(unittest.TestCase):
    """
    Class for tests involving mocu sampling with multivariable X/Y/Theta.
    """

    @classmethod
    def setUpClass(self):

        warnings.filterwarnings("ignore")
        # Prior knowledge: discrete ranges/distributions for (theta,psi)
        theta     = [1,3]
        rho_theta = [1./3 , 2./3]
        Theta     = dict(zip(['theta','rho_theta'] , [theta,rho_theta]))
        psi       = np.linspace(-4,0,101)
        # Prior knowledge: set of experiments X with possible outcomes Y
        # In this example, we do not know rho_j(y_j|(theta_i,x_k)) and must approximate it with MC sampling
        X                      = ['X_1' , 'X_2']
        Y                      = theta
        E                      = dict(zip(['X','Y'] , [X,Y]))
        rho_y_given_theta_exp1 = np.array([ [0.9,0.1] , [0.3,0.7] ])
        rho_y_given_theta_exp2 = np.array([ [0.8,0.2] , [0.1,0.9] ])
        function_X1            = StochasticFunction(theta, rho_theta, Y , rho_y_given_theta_exp1)
        function_X2            = StochasticFunction(theta, rho_theta, Y , rho_y_given_theta_exp2)
        f_experiments          = [function_X1 , function_X2]
        n_sample               = 1000
        self.inputs            = dict(zip( ['Theta','Psi','E','f_experiments','n_sample'] , \
                                           [Theta, psi, E, f_experiments, n_sample] ))
        self.rho_y_given_theta_exp1 = rho_y_given_theta_exp1
        self.rho_y_given_theta_exp2 = rho_y_given_theta_exp2

        # Generate model samples with monte carlo
        y_samples   = []
        y_all       = []
        th_all      = []
        for (i,x) in enumerate(E['X']):
            th_all_i = []
            y_all_i  = []
            y_samples_over_theta  = []
            for (j,th) in enumerate(Theta['theta']):
                y_sample_k = f_experiments[i].forward( th * np.ones(n_sample) )
                y_samples_over_theta.append( y_sample_k )
                for k in range(n_sample):
                    y_all_i.append(y_sample_k[k])
                    th_all_i.append(th)
            y_samples.append( y_samples_over_theta )
            y_all.append( y_all_i )
            th_all.append( th_all_i )
        self.y_samples = y_samples
        self.th_all    = th_all
        self.y_all     = y_all
        
    def test_rho_y_given_theta(self):

        print('vector inputs: compute rho(y | theta,x)')

        # Monte-Carlo to approximate rho(y | theta,x)
        rho_y_given_theta = []
        for (i,x) in enumerate(self.inputs['E']['X']):
            rho_y_given_theta.append( compute_conditional_distribution( self.th_all[i] , \
                                                                        self.y_all[i] , \
                                                                        self.inputs['Theta']['theta'] , \
                                                                        self.inputs['E']['Y'] ) )
        self.rho_y_given_theta = rho_y_given_theta
        self.assertTrue( np.all(np.isclose(rho_y_given_theta[0] , np.array([ [0.9,0.1] , [0.3,0.7] ]) , atol=1e-1 )) )
        self.assertTrue( np.all(np.isclose(rho_y_given_theta[1] , np.array([ [0.8,0.2] , [0.1,0.9] ]) , atol=1e-1 )) )

    def test_compute_probability_theta_given_y(self):

        print('vector inputs: compute rho(theta | y,x)')
        
        rho_theta_given_X1_y , rho_X1_y = compute_probability_theta_given_y( self.rho_y_given_theta_exp1 , \
                                                                             self.inputs['Theta']['rho_theta'] )
        rho_theta_given_X2_y , rho_X2_y = compute_probability_theta_given_y( self.rho_y_given_theta_exp2 , \
                                                                             self.inputs['Theta']['rho_theta'] )
        self.assertTrue( np.all(np.isclose(rho_X1_y , [0.5,0.5])) )
        self.assertTrue( np.all(np.isclose(rho_X2_y , [1/3,2/3])) )
        self.assertTrue( np.all(np.isclose(rho_theta_given_X1_y , [ [3/5,1/15] , [0.4,14/15] ])) )
        self.assertTrue( np.all(np.isclose(rho_theta_given_X2_y , [ [0.8,0.1]  , [0.2,0.9]   ])) )

    def test_psi_ibr(self):

        print('vector inputs: compute psi_ibr(Theta | x,y)')

        rho_theta_given_X1_y = np.array([ [3/5,1/15] , [0.4,14/15] ])
        rho_theta_given_X2_y = np.array([ [0.8,0.1]  , [0.2,0.9]   ])
        rho_theta_given_xi_y = [ rho_theta_given_X1_y , rho_theta_given_X2_y ]
        psi_ibr_X1 , _       = compute_psi_ibr( self.inputs['Psi'] , \
                                                self.inputs['Theta'] , \
                                                self.inputs['E']['X'], \
                                                self.inputs['E']['Y'], \
                                                rho_theta_given_X1_y , \
                                                cost_1parameter_example )
        psi_ibr_X2 , _       = compute_psi_ibr( self.inputs['Psi'] , \
                                                self.inputs['Theta'] , \
                                                self.inputs['E']['X'], \
                                                self.inputs['E']['Y'], \
                                                rho_theta_given_X2_y , \
                                                cost_1parameter_example )
        self.assertTrue( np.all(np.isclose(psi_ibr_X1 , [-9/5 , -43/15] , atol=5e-2)) )
        self.assertTrue( np.all(np.isclose(psi_ibr_X2 , [-1.4 , -2.8]   , atol=5e-2)) )

    def test_omega(self):

        print('scalar inputs: compute E_{theta}[ cost(theta , psi_ibr) ]')

        rho_theta_given_X1_y = np.array([ [3/5,1/15] , [0.4,14/15] ])
        rho_theta_given_X2_y = np.array([ [0.8,0.1]  , [0.2,0.9]   ])
        psi_ibr_X1           = np.array([-9/5 , -43/15])
        psi_ibr_X2           = np.array([-1.4 , -2.8  ])
        omega_X1  = compute_conditional_avg_function( cost_1parameter_example, \
                                                      (self.inputs['Theta']['theta'] , psi_ibr_X1 , [0,1] ), \
                                                      rho_theta_given_X1_y )
        omega_X2  = compute_conditional_avg_function( cost_1parameter_example, \
                                                      (self.inputs['Theta']['theta'] , psi_ibr_X2 , [0,1] ), \
                                                      rho_theta_given_X2_y )
        self.assertTrue( np.all(np.isclose(omega_X1 , [6.76 , 1.78] , atol=5e-2)) )
        self.assertTrue( np.all(np.isclose(omega_X2 , [8.04 , 2.16] , atol=5e-2)) )
        
    def test_mocu_selection(self):
        
        print('vector inputs: main mocu experimental choice function')

        rho_y_given_theta      = [self.rho_y_given_theta_exp1 , self.rho_y_given_theta_exp2]
        mocu_out               = mocu_choose_next_experiment(self.inputs['Theta'],\
                                                             self.inputs['Psi'],\
                                                             self.inputs['E'],\
                                                             rho_y_given_theta,\
                                                             cost_1parameter_example)
        self.assertTrue( mocu_out['x_star'] == 'X_2' )
        self.assertTrue( np.all(np.isclose(mocu_out['avg_omega'] , np.array([4.27 , 4.12]) , atol=1e-2)) )
        

if __name__ == '__main__':
    unittest.main();
