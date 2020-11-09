# Generated with SMOP  0.41
from libsmop import *
# MseOfFilter.m

    
@function
def MseOfFilter(ryy_trace=None,ryx=None,rxx=None,lin_filter=None,*args,**kwargs):
    varargin = MseOfFilter.varargin
    nargin = MseOfFilter.nargin

    mse=ryy_trace - dot(2,sum(sum(multiply(ryx,lin_filter)))) + sum(sum(multiply((dot(lin_filter,rxx)),lin_filter)))
# MseOfFilter.m:2
    return mse
    
if __name__ == '__main__':
    pass
    