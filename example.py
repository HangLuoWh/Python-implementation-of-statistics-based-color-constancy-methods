import numpy as np
from statistics_based.gray_pixels_qian import gray_pixels
import imageio

img = imageio.imread('example.png', 'PNG-FI').astype(np.float32)
mask = np.sum(img, axis=2) != 0
est_illu = gray_pixels(img, mask = None)
print(est_illu)
