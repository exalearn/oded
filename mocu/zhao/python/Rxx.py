# Generated with SMOP  0.41
from libsmop import *
# Rxx.m

    
@function
def Rxx(theta_cell=None,rho=None,sigma=None,*args,**kwargs):
    varargin = Rxx.varargin
    nargin = Rxx.nargin

    rxx_theta=theta_cell[3]
# Rxx.m:2
    rxx=kron(concat([[1 + rho ** 2,dot(2,rho)],[dot(2,rho),1 + rho ** 2]]),rxx_theta) + dot(sigma ** 2,eye(dot(2,size(rxx_theta))))
# Rxx.m:3
    return rxx
    
if __name__ == '__main__':
    pass
    