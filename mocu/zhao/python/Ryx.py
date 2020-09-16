# Generated with SMOP  0.41
from libsmop import *
# Ryx.m

    
@function
def Ryx(theta_cell=None,rho=None,*args,**kwargs):
    varargin = Ryx.varargin
    nargin = Ryx.nargin

    ryx_theta=theta_cell[2]
# Ryx.m:2
    ryx=kron(concat([[1 + rho ** 2,dot(2,rho)],[dot(2,rho),1 + rho ** 2]]),ryx_theta)
# Ryx.m:3
    return ryx
    
if __name__ == '__main__':
    pass
    
    # function ryx = RYX_factor(r, c, theta, bluringT)
#     [c, r] = meshgrid(r, c);
# #     if (size(r)~=size(c))
# #         error('the size must be the same')
# #     end
#     ryx = zeros(size(r, 1), size(c, 2));
#     
#     for m = 1:size(r, 1)
#         for n = 1:size(c, 2)
#             fun = @(x) RYY_factor(r(m, n), x, theta)/bluringT;
#             ryx(m, n) = integral(fun, c(m, n)-bluringT, c(m, n));
#         end
#     end
# end