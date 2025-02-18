from math import cos, sin

import numba
from numba import jit,prange
from .numba_common import piece_wise_integrated, piece_wise_integrated32

import numpy as np

@jit(nopython=True,fastmath=True, parallel=True)
def sA_psf(vol,Ax,psf, angles, reset = True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    psfz,psfx,psfy = psf.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    psf_focus = psfz/2-0.5
    psf_ry = (psfy - 1) / 2
    assert len(angles) == pa, "Projection images and angles don't match"

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

        w_kernel = np.zeros((psfx+2,psfy), dtype='float32')

        for xi in range(nx):
            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                kp = -(xi - xcent) * sinphi + (yi - ycent) * cosphi + psf_focus
                ki = round(kp)
                # extrapolate_Kernel
                ki = max(0,ki)
                ki = min(psfz-1,ki)

                iqx = round(xp)
                xv =  (iqx - xp)
                    
                val1 = piece_wise_integrated32(xv - numba.float32(0.5), a,b,lmax)
                val2 = piece_wise_integrated32(xv + numba.float32(0.5), a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = numba.float32(1.0) - val2

                # make the weighted kernel
                w_kernel*=0
                # flip kernel in x
                w_kernel[:-2,:]+=wx0*psf[ki,:,::-1]
                w_kernel[1:-1,:]+=wx1*psf[ki,:,::-1]
                w_kernel[2:,:]+=wx2*psf[ki,:,::-1]

                kw_x,kw_y = w_kernel.shape
                kr_x, kr_y = (kw_x - 1) // 2, (kw_y - 1) // 2
                
                for kx in range(0,kw_x):
                    ind0 = int(iqx + (kx - kr_x))
                    if (ind0  >= 0 and ind0  < px):
                        kernel = w_kernel[kx,:]
                        for zi in range(kr_y,nh-kr_y): 
                            for ki in range(kw_y):
                                # numpy.dot is slower but a good way to check types
                                Ax[ai, ind0, zi] += vol[xi,yi,zi-kr_y+ki]*kernel[ki]
                    # extend top and bottom with constant boundary condition
                    # by clamping the value index
                        for zi in range(0,kr_y): 
                            for ki in range(kw_y):
                                vzi = max(0,zi-kr_y+ki)
                                Ax[ai, ind0, zi] += vol[xi,yi,vzi]*kernel[ki]
                        for zi in range(nh-kr_y,nh): 
                            for ki in range(kw_y):
                                vzi = min(nh-1,zi-kr_y+ki)
                                Ax[ai, ind0, zi] += vol[xi,yi,vzi]*kernel[ki]

@jit(nopython=True,fastmath=True, parallel=True)
def sA_psf2(vol,Ax,psf, angles, reset = True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    psfz,psfx,psfy = psf.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    psf_focus = psfz/2-0.5
    psf_ry = (psfy - 1) / 2
    assert len(angles) == pa, "Projection images and angles don't match"

    if reset:
        Ax*=0
    for xi in prange(nx):
        for ai in range(pa):
            cosphi = cos(angles[ai])
            sinphi = sin(angles[ai])
            b = (0.5 * abs(cosphi - sinphi))
            a = (0.5 * abs(cosphi + sinphi))
            if b<a:
                a,b=b,a
            lmax = (1 / (a+b))


            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                kp = -(xi - xcent) * sinphi + (yi - ycent) * cosphi + psf_focus
                ki = round(kp)
                # extrapolate_Kernel
                ki = max(0,ki)
                ki = min(psfz-1,ki)

                iqx = round(xp)
                xv =  (iqx - xp)
                    
                val1 = piece_wise_integrated32(xv - numba.float32(0.5), a,b,lmax)
                val2 = piece_wise_integrated32(xv + numba.float32(0.5), a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = numba.float32(1.0) - val2

                # make the weighted kernel
                # w_kernel*=0
                w_kernel = np.zeros((psfx+2,psfy), dtype='float32')

                # flip kernel in x
                w_kernel[:-2,:]+=wx0*psf[ki,:,::-1]
                w_kernel[1:-1,:]+=wx1*psf[ki,:,::-1]
                w_kernel[2:,:]+=wx2*psf[ki,:,::-1]

                kw_x,kw_y = w_kernel.shape
                kr_x, kr_y = (kw_x - 1) // 2, (kw_y - 1) // 2
                
                # numpy.dot is slower but a good way to check types
                for kx in range(0,kw_x):
                    ind0 = int(iqx + (kx - kr_x))
                    if (ind0  >= 0 and ind0  < px):
                        kernel = w_kernel[kx,:]
                        for zi in range(kr_y,nh-kr_y): 
                            for ki in range(kw_y):
                                Ax[ai, ind0, zi] += vol[xi,yi,zi-kr_y+ki]*kernel[ki]
                    # extend top and bottom with constant boundary condition
                    # by clamping the value index
                        for zi in range(0,kr_y): 
                            for ki in range(kw_y):
                                vzi = max(0,zi-kr_y+ki)
                                Ax[ai, ind0, zi] += vol[xi,yi,vzi]*kernel[ki]
                        for zi in range(nh-kr_y,nh): 
                            for ki in range(kw_y):
                                vzi = min(nh-1,zi-kr_y+ki)
                                Ax[ai, ind0, zi] += vol[xi,yi,vzi]*kernel[ki]




@jit(nopython=True,fastmath=True, parallel=True)
def sAT_psf(Ax,vol,psf, angles, reset = True):
    nx,ny,nh = vol.shape
    pa,px,ph = Ax.shape
    psfz,psfx,psfy = psf.shape
    xcent = nx/2-0.5
    ycent = ny/2-0.5
    pcent = px/2-0.5
    psf_focus = psfz/2-0.5
    psf_ry = (psfy - 1) / 2
    assert len(angles) == pa, "Projection images and angles don't match"

    if reset:
        vol*=0

    for xi in prange(nx):
        for ai in range(pa):
            cosphi = cos(angles[ai])
            sinphi = sin(angles[ai])
            b = (0.5 * abs(cosphi - sinphi))
            a = (0.5 * abs(cosphi + sinphi))
            if b<a:
                a,b=b,a
            lmax = (1 / (a+b))

            w_kernel = np.zeros((psfx+2,psfy), dtype='float32')

            for yi in range(ny):
                xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent
                kp = -(xi - xcent) * sinphi + (yi - ycent) * cosphi + psf_focus
                ki = round(kp)
                # extrapolate_Kernel
                ki = max(0,ki)
                ki = min(psfz-1,ki)

                iqx = round(xp)
                xv =  (iqx - xp)
                    
                val1 = piece_wise_integrated32(xv - numba.float32(0.5), a,b,lmax)
                val2 = piece_wise_integrated32(xv + numba.float32(0.5), a,b,lmax)
                wx0 = val1
                wx1 = val2 - val1
                wx2 = numba.float32(1.0) - val2

                # make the weighted kernel
                w_kernel*=0
                # no flip kernel in x (kT)
                w_kernel[:-2,:]+=wx0*psf[ki,:,:]
                w_kernel[1:-1,:]+=wx1*psf[ki,:,:]
                w_kernel[2:,:]+=wx2*psf[ki,:,:]

                kw_x,kw_y = w_kernel.shape
                kr_x, kr_y = (kw_x - 1) // 2, (kw_y - 1) // 2
                
                for kx in range(0,kw_x):
                    ind0 = int(iqx + (kx - kr_x))
                    if (ind0  >= 0 and ind0  < px):
                        kernel = w_kernel[kx,:]
                        for zi in range(kr_y,nh-kr_y): 
                            for ki in range(kw_y):
                                vol[xi,yi,zi]+= Ax[ai, ind0, zi-kr_y+ki]  *kernel[ki]
                    # extend top and bottom with constant boundary condition
                    # by clamping the value index
                        for zi in range(0,kr_y): 
                            for ki in range(kw_y):
                                vzi = max(0,zi-kr_y+ki)
                                vol[xi,yi,zi]+=Ax[ai, ind0, vzi] *kernel[ki]
                        for zi in range(nh-kr_y,nh): 
                            for ki in range(kw_y):
                                vzi = min(nh-1,zi-kr_y+ki)
                                vol[xi,yi,zi]+=Ax[ai, ind0, vzi]*kernel[ki]




