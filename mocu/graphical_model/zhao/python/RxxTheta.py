# Generated with SMOP  0.41
from libsmop import *
# RxxTheta.m

    
@function
def RxxTheta(r=None,c=None,theta=None,blurringT=None,*args,**kwargs):
    varargin = RxxTheta.varargin
    nargin = RxxTheta.nargin

    #     [c, r] = meshgrid(c, r);
#     if (size(t1)~=size(t2))
#         error('the size must be the same')
#     end
    theta
    rxx_theta=zeros(size(r,1),size(c,2))
# RxxTheta.m:7
    fun=lambda x=None,y=None: RyyTheta(x,y,theta) / (blurringT ** 2)
# RxxTheta.m:8
    for m in arange(1,size(r,1)).reshape(-1):
        for n in arange(1,size(c,2)).reshape(-1):
            rxx_theta[m,n]=integral2(fun,r(m) - blurringT,r(m),c(n) - blurringT,c(n),'AbsTol',0.0001)
# RxxTheta.m:11
    
    return rxx_theta
    
if __name__ == '__main__':
    pass
    
    #decomposite into r+c and r-c