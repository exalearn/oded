# Generated with SMOP  0.41
from libsmop import *
# RyyTrace.m

    
@function
def RyyTrace(r=None,rho=None,theta_cell=None,*args,**kwargs):
    varargin = RyyTrace.varargin
    nargin = RyyTrace.nargin

    rhomat=concat([[1 + rho ** 2,dot(2,rho)],[dot(2,rho),1 + rho ** 2]])
# RyyTrace.m:2
    ryy_trace=dot(sum(RyyTheta(r,r,theta_cell[1])),trace(rhomat))
# RyyTrace.m:3
    return ryy_trace
    
if __name__ == '__main__':
    pass
    