from math import cos, sin

import numba
from numba import jit,prange
from .numba_common import piece_wise_integrated, piece_wise_integrated32

@jit(nopython=True,fastmath=True, parallel=False, error_model='numpy')
def A_bl(vol,Ax,angles, reset= True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    assert len(angles) == pa, (f"Projection images ({pa})"
                          f"and angles {len(angles)} don't match")
    
    if reset:
        Ax*=0
        
    for ai in prange(pa):
        cosphi = cos(angles[ai])
        sinphi = sin(angles[ai])
        b = (0.5 * abs(cosphi - sinphi))
        a = (0.5 * abs(cosphi + sinphi))
        if b<a:
            a,b=b,a
        lmax = (1 / (a+b))

        for xi in range(nx):
            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                iqx = round(xp)
                xv = (iqx - xp)
                    
                val1 = piece_wise_integrated(xv - 0.5, a,b,lmax)
                val2 = piece_wise_integrated(xv + 0.5, a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = 1.0 - val2

                if (iqx - 1 >= 0 and iqx - 1 < px):
                    for zi in range(nh): 
                        Ax[ai,iqx - 1,zi]+=vol[xi,yi,zi] * wx0
                if (iqx  >= 0 and iqx  < px):
                    for zi in range(nh): 
                        Ax[ai,iqx,zi]+=vol[xi,yi,zi] * wx1
                if (iqx + 1 >= 0 and iqx + 1 < px):
                    for zi in range(nh): 
                        Ax[ai,iqx+1,zi]+=vol[xi,yi,zi] * wx2

@jit(nopython=True,fastmath=True, parallel=True)
def sA_bl(vol,Ax,angles, reset= True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    assert len(angles) == pa, (f"Projection images ({pa})"
                          f"and angles {len(angles)} don't match")
    
    if reset:
        Ax*=0
        
    for ai in prange(pa):
        cosphi = cos(angles[ai])
        sinphi = sin(angles[ai])
        b = (0.5 * abs(cosphi - sinphi))
        a = (0.5 * abs(cosphi + sinphi))
        if b<a:
            a,b=b,a
        lmax = (1 / (a+b))

        for xi in range(nx):
            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                iqx = round(xp)
                xv = (iqx - xp)
                    
                val1 = piece_wise_integrated32(xv - numba.float32(0.5), a,b,lmax)
                val2 = piece_wise_integrated32(xv + numba.float32(0.5), a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = numba.float32(1.0) - val2

                if (iqx - 1 >= 0 and iqx - 1 < px):
                    # Ax[ai,iqx - 1,:]+=vol[xi,yi,:] * wx0 is slower
                    for zi in range(nh): 
                        Ax[ai,iqx - 1,zi]+=vol[xi,yi,zi] * wx0
                if (iqx  >= 0 and iqx  < px):
                    for zi in range(nh): 
                        Ax[ai,iqx,zi]+=vol[xi,yi,zi] * wx1
                if (iqx + 1 >= 0 and iqx + 1 < px):
                    for zi in range(nh): 
                        Ax[ai,iqx+1,zi]+=vol[xi,yi,zi] * wx2



@jit(nopython=True,fastmath=True, parallel=True)
def AT_bl(Ax,vol,angles, reset= True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    assert len(angles) == pa, (f"Projection images ({pa})"
                          f"and angles {len(angles)} don't match")
    
    if reset:
        vol*=0
        
    for ai in prange(pa):
        cosphi = cos(angles[ai])
        sinphi = sin(angles[ai])
        b = (0.5 * abs(cosphi - sinphi))
        a = (0.5 * abs(cosphi + sinphi))
        if b<a:
            a,b=b,a
        lmax = (1 / (a+b))

        for xi in range(nx):
            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                iqx = round(xp)
                xv = (iqx - xp)
                    
                val1 = piece_wise_integrated(xv - 0.5, a,b,lmax)
                val2 = piece_wise_integrated(xv + 0.5, a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = 1.0 - val2

                if (iqx - 1 >= 0 and iqx - 1 < px):
                    for zi in range(nh): 
                        vol[xi,yi,zi]+=Ax[ai,iqx - 1,zi]* wx0
                if (iqx  >= 0 and iqx  < px):
                    for zi in range(nh): 
                        vol[xi,yi,zi]+=Ax[ai,iqx,zi] * wx1
                if (iqx + 1 >= 0 and iqx + 1 < px):
                    for zi in range(nh): 
                        vol[xi,yi,zi]+=Ax[ai,iqx+1,zi] * wx2

@jit(nopython=True,fastmath=True, parallel=True)
def sAT_bl(Ax,vol,angles, reset= True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    assert len(angles) == pa, (f"Projection images ({pa})"
                          f"and angles {len(angles)} don't match")
    
    if reset:
        vol*=0
        
    for ai in prange(pa):
        cosphi = cos(angles[ai])
        sinphi = sin(angles[ai])
        b = (0.5 * abs(cosphi - sinphi))
        a = (0.5 * abs(cosphi + sinphi))
        if b<a:
            a,b=b,a
        lmax = (1 / (a+b))

        for xi in range(nx):
            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                iqx = round(xp)
                xv = (iqx - xp)
                    
                val1 = piece_wise_integrated32(xv - numba.float32(0.5), a,b,lmax)
                val2 = piece_wise_integrated32(xv + numba.float32(0.5), a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = numba.float32(1.0) - val2

                if (iqx - 1 >= 0 and iqx - 1 < px):
                    # Ax[ai,iqx - 1,:]+=vol[xi,yi,:] * wx0 is slower
                    for zi in range(nh): 
                        vol[xi,yi,zi]+=Ax[ai,iqx - 1,zi] * wx0
                if (iqx  >= 0 and iqx  < px):
                    for zi in range(nh): 
                        vol[xi,yi,zi]+=Ax[ai,iqx,zi] * wx1
                if (iqx + 1 >= 0 and iqx + 1 < px):
                    for zi in range(nh): 
                        vol[xi,yi,zi]+=Ax[ai,iqx+1,zi] * wx2