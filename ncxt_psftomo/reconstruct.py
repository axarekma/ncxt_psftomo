import numpy  as np
from tqdm.auto import trange

from .numba_bl import sA_bl, sAT_bl
from .numba_psf import sA_psf, sAT_psf

from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

def CG_BL(x, y, angles, n_iter, chatty = False,disable_tqdm=True, oracle = None,stop_at_best=True):
    _y = np.zeros(y.shape, dtype = x.dtype)
    _r = np.zeros(y.shape, dtype = x.dtype)
    _q = np.zeros(y.shape, dtype = x.dtype)
    _s = np.zeros(x.shape, dtype = x.dtype)

    sA_bl(x, _y, angles)
    _r = y - _y
    sAT_bl(_r,_s, angles)
    gamma0 = np.sum(_s * _s)
    _p = _s

    if oracle is not None:
        psnr = []

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

        if oracle is not None:
            psnr.append(peak_signal_noise_ratio(x, oracle, data_range=np.max(oracle)))
            if np.argmax(psnr) == i:
                x_best = np.copy(x)
            elif np.argmax(psnr) < i-2 and stop_at_best:
                return x_best, psnr

    if oracle is not None:
        return x, psnr
    return x

def MLEM_BL(x, y, angles, n_iter, chatty = False,disable_tqdm=True, oracle = None, stop_at_best = True):
    EPS = 1e-9
    y_clip = np.clip(y,EPS,np.inf)

    coeff = np.zeros_like(x)
    norm = np.zeros_like(x)
    y_est = np.zeros_like(y)

    y_weights = np.ones_like(y)
    sAT_bl(y_weights, norm, angles)

    if oracle is not None:
        psnr = []

    bar = trange(n_iter, leave=False, disable=disable_tqdm)
    for i in bar:
        sA_bl(x, y_est, angles)
        ratio = y_clip/(EPS+y_est)
        sAT_bl(ratio, coeff, angles)
        x*=(coeff/norm)

        if oracle is not None:
            psnr.append(peak_signal_noise_ratio(x, oracle, data_range=np.max(oracle)))
            if np.argmax(psnr) == i:
                x_best = np.copy(x)
            elif np.argmax(psnr) < i-2 and stop_at_best:
                return x, psnr

    if oracle is not None:
        return x, psnr
    return x

# def MLEM_BL(x, y, angles, n_iter, blocks = 1,exponent = 1.0, chatty = False,
#             disable_tqdm=True, oracle = None,stop_at_best= True):
#     EPS = 1e-9

#     angles_blocks = [angles[i::blocks] for i in range(blocks)]
#     y_blocks = [np.clip(y[i::blocks],EPS,np.inf) for i in range(blocks)]

#     coeff = np.zeros_like(x)
#     norm = np.zeros_like(x)
    
#     y_est = [ np.zeros_like(el) for el in y_blocks]
#     y_weight = [ np.ones_like(el) for el in y_blocks]
#     norm = [ np.ones_like(x) for i in range(blocks)]

#     for yw,n,ang in zip(y_weight,norm,angles_blocks): 
#         sAT_bl(yw, n, ang)

#     if oracle is not None:
#         psnr = []
    
#     bar = trange(n_iter, leave=False, disable=disable_tqdm)
#     for i in bar:
#         x_static = np.copy(x)
#         # loop

#         for bi in range(blocks):
#             sA_bl(x_static, y_est[bi], angles_blocks[bi])
#             ratio = y_blocks[bi]/(EPS+y_est[bi])
#             sAT_bl(ratio, coeff, angles_blocks[bi])
#             x*=(coeff/norm[bi])**(exponent/blocks)

#         if oracle is not None:
#             psnr.append(peak_signal_noise_ratio(x, oracle, data_range=np.max(oracle)))
#             if np.argmax(psnr) == i:
#                 x_best = np.copy(x)
#             elif np.argmax(psnr) < i-2 and stop_at_best:
#                 return x, psnr

#     if oracle is not None:
#         return x, psnr
#     return x

def MLEM_BL_upd(x, y, angles, n_iter, blocks = 1,exponent = 1.0, 
                chatty = False,disable_tqdm=True, oracle = None,stop_at_best = True):
    EPS = 1e-9

    angles_blocks = [angles[i::blocks] for i in range(blocks)]
    y_blocks = [np.clip(y[i::blocks],EPS,np.inf) for i in range(blocks)]

    coeff = np.zeros_like(x)
    norm = np.zeros_like(x)
    
    y_est = [ np.zeros_like(el) for el in y_blocks]
    y_weight = [ np.ones_like(el) for el in y_blocks]
    norm = [ np.ones_like(x) for i in range(blocks)]

    for yw,n,ang in zip(y_weight,norm,angles_blocks): 
        sAT_bl(yw, n, ang)

    if oracle is not None:
        psnr = []
    
    bar = trange(n_iter, leave=False, disable=disable_tqdm)
    for i in bar:
        
        # loop

        for bi in range(blocks):
            sA_bl(x, y_est[bi], angles_blocks[bi])
            ratio = y_blocks[bi]/(EPS+y_est[bi])
            sAT_bl(ratio, coeff, angles_blocks[bi])
            x*=(coeff/norm[bi])**(exponent)
            # print(f'times C {absc:.2f} M {absmul:.2f} Xs {absxs:.2f} :: X {absx0:.2f}--> {np.linalg.norm(x):.2f}')


        if oracle is not None:
            psnr.append(peak_signal_noise_ratio(x, oracle, data_range=np.max(oracle)))
            if np.argmax(psnr) == i:
                x_best = np.copy(x)
            elif np.argmax(psnr) < i-2 and stop_at_best:
                return x, psnr

    if oracle is not None:
        return x, psnr
    return x


def CG_PSF(x, y,psf, angles, n_iter, chatty = False,disable_tqdm = True, oracle = None, stop_at_best = True):
    _y = np.zeros(y.shape, dtype = x.dtype)
    _r = np.zeros(y.shape, dtype = x.dtype)
    _q = np.zeros(y.shape, dtype = x.dtype)
    _s = np.zeros(x.shape, dtype = x.dtype)

    sA_psf(x, _y,psf, angles)
    _r = y - _y
    sAT_psf(_r,_s,psf, angles)
    gamma0 = np.sum(_s * _s)
    _p = _s

    if oracle is not None:
        psnr = []

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

        if oracle is not None:
            psnr.append(peak_signal_noise_ratio(x, oracle, data_range=np.max(oracle)))
            if np.argmax(psnr) == i:
                x_best = np.copy(x)
            elif np.argmax(psnr) < i-2 and stop_at_best:
                return x_best, psnr

    if oracle is not None:
        return x, psnr
    return x

def MLEM_PSF(x, y, psf, angles, n_iter, chatty = False,disable_tqdm=True, oracle = None, stop_at_best = True):
    EPS = 1e-9
    y_clip = np.clip(y,EPS,np.inf)

    coeff = np.zeros_like(x)
    norm = np.zeros_like(x)
    y_est = np.zeros_like(y)

    y_weights = np.ones_like(y)
    sAT_psf(y_weights, norm, psf,angles)

    if oracle is not None:
        psnr = []

    bar = trange(n_iter, leave=False, disable=disable_tqdm)
    for i in bar:
        sA_psf(x, y_est,psf, angles)
        ratio = y_clip/(EPS+y_est)
        sAT_psf(ratio, coeff,psf, angles)
        x*=(coeff/norm)

        if oracle is not None:
            psnr.append(peak_signal_noise_ratio(x, oracle, data_range=np.max(oracle)))
            if np.argmax(psnr) == i:
                x_best = np.copy(x)
            elif np.argmax(psnr) < i-2 and stop_at_best:
                return x_best, psnr

    if oracle is not None:
        return x, psnr
    return x