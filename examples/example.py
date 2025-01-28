import numpy as np
import matplotlib.pyplot as plt

import ncxt_psftomo
from ncxt_psftomo.reconstruct import CG_BL, CG_PSF

from pathlib import Path
print("File      Path:", Path(__file__))
currfolder = Path(__file__).parents[0]
datafolder = Path(__file__).parents[1] /'data'

dtype = 'float32'
phantom = ncxt_psftomo.read_write_mrc.read_mrc(datafolder/'phantom.mrc').astype(dtype)

pad = 16
phantom= phantom[pad:-pad,pad:-pad,pad:-pad]
R = phantom.shape[0]//2

resolution, dof,psf_r = 3,50,3
psf = ncxt_psftomo.psf_utils.psf_volume(resolution,dof,resolution,2*dof)

angles = np.linspace(0,np.pi,90,endpoint = False)
projections = np.zeros((len(angles),phantom.shape[0],phantom.shape[1]), dtype = dtype)
projections_bl = np.zeros((len(angles),phantom.shape[0],phantom.shape[1]), dtype = dtype)
ncxt_psftomo.forward_bl(1.0*phantom, projections_bl, angles)
ncxt_psftomo.forward(1.0*phantom, projections,psf, angles)


print('Ideal rec...')
rec_test = np.zeros(phantom.shape, dtype = dtype)
rec_test = CG_BL(rec_test, projections_bl, angles, 10, chatty=False, disable_tqdm=False)

print('BL approx...')
rec_bl = np.zeros(phantom.shape, dtype = dtype)
rec_bl = CG_BL(rec_bl, projections, angles, 30, chatty=False, disable_tqdm=False)

print('PSF inversion...')
rec_psf = np.zeros(phantom.shape, dtype = dtype)
rec_psf = CG_PSF(rec_psf, projections,psf, angles, 30, chatty=False, disable_tqdm=False)

from skimage.metrics import peak_signal_noise_ratio
f, axes = plt.subplots(ncols = 4, figsize = (13,5))
axes[0].imshow(phantom[:,:,R])
axes[1].imshow(rec_test[:,:,R])
axes[2].imshow(rec_bl[:,:,R])
axes[3].imshow(rec_psf[:,:,R])

axes[0].set_title('Phantom')
axes[1].set_title(f' Ideal ({peak_signal_noise_ratio(rec_test,phantom, data_range=np.max(phantom)):.2f})')
axes[2].set_title(f' BL ({peak_signal_noise_ratio(rec_bl,phantom, data_range=np.max(phantom)):.2f})')
axes[3].set_title(f' PSF ({peak_signal_noise_ratio(rec_psf,phantom, data_range=np.max(phantom)):.2f})')
plt.savefig(currfolder/'example_PSNR', bbox_inches = 'tight')