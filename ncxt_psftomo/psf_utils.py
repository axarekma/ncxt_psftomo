import numpy as np
from scipy.special import jv
from imageio import imread


def Lommel(u, v, n):
    U = u * 0
    for m in range(50):
        U += (-1)**m * (u / v)**(n + 2 * m) * jv(n + 2 * m, v)
    return U


def PSF_uv_vec(u, v):
    sel_uv = (np.abs(u) + np.abs(v) == 0)
    sel_u = (u == 0) ^ sel_uv
    sel_v = (v == 0) ^ sel_uv
    sel = ~(sel_u + sel_v + sel_uv)

    h = np.zeros(u.shape)

    h[sel] = (2 / u[sel])**2 * (
        Lommel(u[sel], v[sel], 1)**2 + Lommel(u[sel], v[sel], 2)**2)

    h[sel_u] = (2 * jv(1, v[sel_u]) / v[sel_u])**2
    h[sel_v] = (np.sin(u[sel_v] / 4) / (u[sel_v] / 4))**2
    h[sel_uv] = 1
    return h


def PSF_rz_vec(r, z, lamb, NA):
    u = 2 * np.pi / lamb * NA**2 * z
    v = 2 * np.pi / lamb * NA * r

    return PSF_uv_vec(u, v)


def geth_R_shift(dx, dz, lambd, NA, L, R, f):
    rayleigh = 0.610 * lambd / NA / dx
    focal_point = f * L / 2
    r = dx * np.linspace(-R, R, 2 * R + 1)
    z = dz * np.linspace(0, L, L) - focal_point
    zz, rr = np.meshgrid(z, r)
    print(rr.shape)
    h = PSF_rz_vec(rr, zz, lambd, NA)

    return h, rayleigh


def psf_slice(resolution, depth_of_field, R, L):
    NA = resolution / (0.610 * depth_of_field)
    wavelength = depth_of_field * NA**2

    r = np.linspace(-R, R, 2 * R + 1)
    z = np.linspace(0, L, L) - L / 2
    zz, rr = np.meshgrid(z, r)
    h = PSF_rz_vec(rr, zz, wavelength, NA)
    # h /= np.sum(h, 0)

    return h

def psf_volume(resolution, depth_of_field, R, L):
    NA = resolution / (0.610 * depth_of_field)
    wavelength = depth_of_field * NA**2

    r = np.linspace(-R, R, 2 * R + 1)
    z = np.linspace(0, L, L) - L / 2
    zz, rr1,rr2 = np.meshgrid(z, r,r,indexing='ij')
    rr = np.sqrt(rr1**2+rr2**2)
    h = PSF_rz_vec(rr, zz, wavelength, NA)
    for h_slice in h:
        h_slice/=np.sum(h_slice)


    return h
