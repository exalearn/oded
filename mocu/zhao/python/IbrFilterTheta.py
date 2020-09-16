# Generated with SMOP  0.41
from libsmop import *
# IbrFilterTheta.m

    
@function
def IbrFilterTheta(rho=None,theta_dispara=None,r=None,c=None,blurringT=None,sigma_=None,*args,**kwargs):
    varargin = IbrFilterTheta.varargin
    nargin = IbrFilterTheta.nargin

    #theta is distributed among (1-Theta/2, 1+Theta/2) with pdf 1/Theta.
    theta_center=5
# IbrFilterTheta.m:3
    
    upperbound=theta_center + theta_dispara / 2
# IbrFilterTheta.m:6
    lowerbound=theta_center - theta_dispara / 2
# IbrFilterTheta.m:7
    thetapdf=lambda theta=None: 1 / theta_dispara
# IbrFilterTheta.m:8
    #     N = 5000;
    fun_yx=lambda theta=None: dot(RyxTheta(r,c,theta,blurringT),thetapdf(theta))
# IbrFilterTheta.m:11
    fun_xx=lambda theta=None: dot(RxxTheta(r,c,theta,blurringT),thetapdf(theta))
# IbrFilterTheta.m:12
    #     fun3 = @(theta) RyxTheta(51, 43, theta, blurringT)./theta_dispara;
#     integral(fun3, lowerbound, upperbound,  'ArrayValued',true);
#     ryx_theta_dis = integral(fun_yx, lowerbound, upperbound);
    ryx_theta_dis=integral(fun_yx,lowerbound,upperbound,'ArrayValued',true,'AbsTol',0.001)
# IbrFilterTheta.m:16
    rxx_theta_dis=integral(fun_xx,lowerbound,upperbound,'ArrayValued',true,'AbsTol',0.001)
# IbrFilterTheta.m:17
    rhomat=concat([[1 + rho ** 2,dot(2,rho)],[dot(2,rho),1 + rho ** 2]])
# IbrFilterTheta.m:19
    ryx_IBR_theta=kron(rhomat,ryx_theta_dis)
# IbrFilterTheta.m:21
    rxx_IBR_theta=kron(rhomat,rxx_theta_dis) + dot(sigma_ ** 2,eye(dot(2,size(rxx_theta_dis))))
# IbrFilterTheta.m:22
    IBR_filter_theta=ryx_IBR_theta / rxx_IBR_theta
# IbrFilterTheta.m:24
    return IBR_filter_theta
    
if __name__ == '__main__':
    pass
    