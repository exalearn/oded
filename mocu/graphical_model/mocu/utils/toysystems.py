import numpy as np
from scipy.integrate import ode, odeint
import sdeint

def make_f(a,b,c):
    def f(y,t):
        return -(y-c) * (y-a) * (y-b)
    return f

def make_noise_function(theta):
    def g(y,t):
        return theta
    return g

def make_rhs_full_system(a,b,k,c,lam):    
    def rhs_full_system(y,t):
        C      = c(a,b,k,y[0])
        y1_dot = lam[0] * (y[0] - 1)
        y2_dot = lam[1] * (y[1] - C) * (y[1] - a) * (y[1] - b)
        return [y1_dot , y2_dot]
    return rhs_full_system
