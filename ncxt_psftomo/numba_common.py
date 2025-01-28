import numba
from numba import jit,prange
from math import cos, sin
import numpy as np

@jit("float32(float32,float32,float32,float32)",nopython=True,fastmath=True)
def piece_wise_integrated32(x, a,b,y_max):
    if x <= -b:
        return  numba.float32(0)
    if x >= b:
        return  numba.float32(1)
    
    if (x >= -a and x <= a):
        return numba.float32(0.5) * y_max * (b + a + numba.float32(2)*x); 
    if (x < -a):
        return numba.float32(0.5) * y_max * (x + b) * (x + b) / (b - a)
    if (x > a):
        return numba.float32(0.5) * y_max * (b - x) * (b - x) / (a - b) + numba.float32(1.0)
    return numba.float32(0)

@jit(nopython=True,fastmath=True, ) 
def piece_wise_integrated(x, a,b,y_max):
    if x <= -b:
        return  0
    if x >= b:
        return  1
    
    if (x >= -a and x <= a):
        return 0.5 * y_max * (b + a + 2*x); 
    if (x < -a):
        return 0.5 * y_max * (x + b) * (x + b) / (b - a)
    if (x > a):
        return 0.5 * y_max * (b - x) * (b - x) / (a - b) + 1.0
    return 0

