import numpy as np
from methods.statistics_based.gray_pixels_qian import gray_pixels
import imageio

img = imageio.imread('00_0011.png', 'PNG-FI').astype(np.float32)
mask = np.sum(img, axis=2) != 0
tmp = gray_pixels(img, mask = None)
print(tmp)