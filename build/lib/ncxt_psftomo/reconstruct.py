import numpy  as np
from tqdm.auto import trange

from .numba_bl import sA_bl, sAT_bl
from .numba_psf import sA_psf, sAT_psf

def CG_BL(x, y, angles, n_iter, chatty = True,disable_tqdm=True):
    _y = np.zeros(y.shape, dtype = x.dtype)
    _r = np.zeros(y.shape, dtype = x.dtype)
    _q = np.zeros(y.shape, dtype = x.dtype)
    _s = np.zeros(x.shape, dtype = x.dtype)

    sA_bl(x, _y, angles)
    _r = y - _y
    sAT_bl(_r,_s, angles)
    gamma0 = np.sum(_s * _s)
    _p = _s

    bar = trange(n_iter, leave=False, disable=disable_tqdm)
    for i in bar:
        sA_bl(_p, _q,angles)
        alpha = gamma0 / np.sum(_q * _q)
        x += alpha * _p
        _r -= alpha * _q

        sAT_bl(_r,_s, angles)
        gamma = np.sum(_s * _s)
        beta = gamma / gamma0
        gamma0 = gamma       
        _p = _s + beta * _p 
        if chatty:      
            print(f'Iter {i} alpha {alpha:.2e} resnorm {np.sqrt(np.sum(_r*_r)):.2e}')
    return x

def CG_PSF(x, y,psf, angles, n_iter, chatty = True,disable_tqdm = True):
    _y = np.zeros(y.shape, dtype = x.dtype)
    _r = np.zeros(y.shape, dtype = x.dtype)
    _q = np.zeros(y.shape, dtype = x.dtype)
    _s = np.zeros(x.shape, dtype = x.dtype)

    sA_psf(x, _y,psf, angles)
    _r = y - _y
    sAT_psf(_r,_s,psf, angles)
    gamma0 = np.sum(_s * _s)
    _p = _s

    bar = trange(n_iter, leave=False, disable=disable_tqdm)
    for i in bar:
        sA_psf(_p, _q,psf,angles)
        alpha = gamma0 / np.sum(_q * _q)
        x += alpha * _p
        _r -= alpha * _q

        sAT_psf(_r,_s ,psf, angles)
        gamma = np.sum(_s * _s)
        beta = gamma / gamma0
        gamma0 = gamma       
        _p = _s + beta * _p        
        if chatty:      
            print(f'Iter {i} alpha {alpha:.2e} resnorm {np.sqrt(np.sum(_r*_r)):.2e}')
    return x