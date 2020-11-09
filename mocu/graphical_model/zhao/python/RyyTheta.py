# Generated with SMOP  0.41
from libsmop import *
# RyyTheta.m

    
@function
def RyyTheta(r=None,c=None,theta=None,*args,**kwargs):
    varargin = RyyTheta.varargin
    nargin = RyyTheta.nargin

    #return a matrix of size(t1) = size(t2)
#the autocorrelation is calculated by RYY = kron([1+rho^2, 2*rho; 2*rho, 1+rho^2], ryy);
#     [c, r] = meshgrid(c, r);
    ryy=multiply(1.0 / (dot(2,theta)),(exp(dot(dot(0.01,theta),(r + c))) - exp(dot(dot(0.01,theta),abs(r - c)))))
# RyyTheta.m:5
    ryy=multiply(multiply(ryy,(r > 0)),(c > 0))
# RyyTheta.m:6
    return ryy
    
if __name__ == '__main__':
    pass
    