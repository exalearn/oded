# Generated with SMOP  0.41
from libsmop import *
# IbrFilterRho.m

    
@function
def IbrFilterRho(theta_cell=None,beta_=None,sigma_=None,*args,**kwargs):
    varargin = IbrFilterRho.varargin
    nargin = IbrFilterRho.nargin

    # given a theta value, covariance is kron product of theta part and rho
    #part, here the IBR is over rho, which is beta distributed
    #theta cell is a temporary setting to avoid repeat calculation
    ryx_theta=theta_cell[2]
# IbrFilterRho.m:5
    rxx_theta=theta_cell[3]
# IbrFilterRho.m:6
    betamat=concat([[1 + 1 / (dot(2,beta_) + 1),0],[0,1 + 1 / (dot(2,beta_) + 1)]])
# IbrFilterRho.m:7
    ryx_IBR_rho=kron(betamat,ryx_theta)
# IbrFilterRho.m:8
    rxx_IBR_rho=kron(betamat,rxx_theta) + dot(sigma_ ** 2,eye(dot(2,size(ryx_theta))))
# IbrFilterRho.m:9
    IBR_filter_rho=ryx_IBR_rho / rxx_IBR_rho
# IbrFilterRho.m:11
    return IBR_filter_rho
    
if __name__ == '__main__':
    pass
    