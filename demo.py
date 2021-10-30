from scipy import io
import numpy as np
from PIL import Image
import time
from COCNMF import *

# Setting
data_mat = io.loadmat('data.mat')
seed_mat = io.loadmat('seed.mat')

I_REF = data_mat['I_REF']
Ym = data_mat['Ym']
Yh = data_mat['Yh']
D = data_mat['D']
ratio = data_mat['ratio'][0][0]
symvar = data_mat['symvar'][0][0]
N = data_mat['N'][0][0]
seed = seed_mat['seed'][0]
seed = seed[0]

K = gaussion2D([ratio, ratio], symvar)

# Generate Yh data
rows_h, cols_h, bands_h = Yh.shape
Yh_gen = Yh.reshape(-1, bands_h, order='F').T
SNR_Yh = 35
varianc_Yh = np.mean(np.square(Yh_gen), axis=1) * (10 ** (-SNR_Yh / 10))
np.random.seed(seed)
Yh_gen = Yh_gen + np.diag(varianc_Yh) @ np.random.normal(size=Yh_gen.shape)
Yh_gen = Yh_gen.T.reshape(Yh.shape, order='F')

# Generate Ym data
rows_m, cols_m, bands_m = Ym.shape
Ym_gen = Ym.reshape(-1, bands_m, order='F').T
SNR_Ym = 30
varianc_Ym = np.mean(np.square(Ym_gen), axis=1) * (10 ** (-SNR_Ym / 10))
np.random.seed(seed)
Ym_gen = Ym_gen + np.diag(varianc_Ym) @ np.random.normal(size=Ym_gen.shape)
Ym_gen = Ym_gen.T.reshape(Ym.shape, order='F')

# CO-CNMF
time_start = time.time()
Z_fused = ConvOptiCNMF(Yh_gen, Ym_gen, N, D, K) # Convex Optimization based Coupled NMF (CO-CNMF)
time_end = time.time()
time_COCNMF = time_end - time_start

# Plot & Display
band_set = [60, 24, 12] # RGB bands

img_REF = I_REF[:, :, band_set]
img_REF = Image.fromarray((normColor(img_REF) * 255).astype(np.uint8))
img_REF.save('I_REF.png')
print('(a) (Ground Truth) Reference Image: I_REF.png')

small_img = normColor(Yh_gen)[:, :, band_set]
scaled_img = np.ndarray([small_img.shape[0] * ratio, small_img.shape[1] * ratio, 3])
for i in range(3):
    scaled_img[:, :, i] = np.kron(small_img[:, :, i], np.ones([ratio, ratio]))
img_Yh = Image.fromarray((scaled_img * 255).astype(np.uint8))
img_Yh.save('Yh.png')
print('(b) Low-resolution Input Image: Yh.png')

img_Ym = Ym_gen[:, :, [2, 1, 0]]
img_Ym = Image.fromarray((normColor(img_Ym) * 255).astype(np.uint8))
img_Ym.save('Ym.png')
print('(c) High-resolution Input Image: Ym.png')

img_Z = Z_fused[:, :, band_set]
img_Z = Image.fromarray((normColor(img_Z) * 255).astype(np.uint8))
img_Z.save('Z_fused.png')
print('(d) Super-resolved Output Image: Z_fused.png')

print()

print('CO-CNMF performance:')
bands = I_REF.shape[2]
ref = I_REF.reshape(-1, bands, order='F')
tar = Z_fused.reshape(-1, bands, order='F')
msr = np.mean(np.power(ref - tar, 2), axis=0)
max2= np.power(np.max(tar, axis=0), 2)
PSNR = np.mean(10 * np.log10(np.divide(max2, msr)))
print(f'PSNR: {PSNR:.2f} (dB)')
print(f'TIME: {time_COCNMF:.2f} (sec.)')