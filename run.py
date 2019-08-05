from skimage import data
from skimage.transform import resize
import psfrec
import numpy as np
import matplotlib.pyplot as plt
from psfrec import psf_utils
from tqdm import tqdm
from functools import reduce
from scipy.signal.windows import tukey


def window(shape, func, **kwargs):
    vs = [func(l, **kwargs) for l in shape]
    return reduce(np.multiply, np.ix_(*vs))


def run():
    L = 64
    na = int(np.pi * L / 2)
    Lp = int(np.sqrt(2) * L) + 3
    print(np.sqrt(2) * L, Lp)
    psf_z = Lp

    res, dof = 5, L // 2
    psf_r = 5
    psf = psf_utils.get_psf(res, dof, psf_r, psf_z)

    x = resize(data.camera(), (L, L), anti_aliasing=True, mode='reflect')
    x *= window(x.shape, tukey)

    angles = np.linspace(0, np.pi, na)

    y = np.zeros((Lp, na))
    psfrec.forward_2d(x, y, psf, angles)

    # init vars

    resnorm = []
    x_n = x * 0.0
    y_n = y * 0.0
    psf_n = psf_utils.get_psf(1, psf_z, psf_r, psf_z)
    # psf_n = 1.0 * psf
    r = y * 0.0
    q = y * 0.0
    s = x * 0.0

    psf_diff = psf_n * 0 + 1e-6

    print()
    # for n in tqdm(range(50)):
    for n in range(100):
        y_n *= 0
        s *= 0
        q *= 0
        psf_diff *= 0.0

        gamma = 1e-4

        psfrec.forward_2d(x_n, y_n, psf_n, angles)
        r = y - y_n
        resnorm.append(np.linalg.norm(r))

        psfrec.backward_2d(s, r, psf_n, angles)
        psfrec.forward_2d(s, q, psf_n, angles)

        alpha = np.sum(s * s) / np.sum(q * q)
        x_n += alpha * s

        # y_n *= 0
        # psfrec.forward_2d(x_n, y_n, psf_n, angles)

        psfrec.grad_psf_2d(x_n, r, psf_diff, angles)
        psf_n += gamma * psf_diff

        # print(np.mean(np.abs(gamma * psf_diff / reldiff) / np.abs(psf_n)))
        psf_n /= np.sum(psf_n)
        psf_n *= np.sum(psf)

        x_n[x_n < 0] *= 0.5
        psf_n[psf_n < 0] *= 0.5

    print('DONE')
    print(f'{np.min(x_n)} {np.max(x_n)}')
    print(f'{np.min(psf_n)} {np.max(psf_n)}')
    print(f'{np.min(psf_diff)} {np.max(psf_diff)}')
    plt.figure(figsize=(15, 13))
    plt.subplot(231)
    plt.imshow(y)
    plt.subplot(234)
    plt.imshow(y_n)

    plt.subplot(232)
    plt.imshow(x)
    plt.subplot(235)
    plt.imshow(x_n)

    plt.subplot(233)
    conc_psf = np.concatenate(
        (psf / np.max(np.abs(psf)), psf_n / np.max(np.abs(psf_n)),
         psf_diff / np.max(np.abs(psf_diff))), 0)
    plt.imshow(conc_psf)
    plt.subplot(236)
    plt.semilogy(resnorm)
    # plt.show()
    plt.savefig('res')


if __name__ == '__main__':
    run()