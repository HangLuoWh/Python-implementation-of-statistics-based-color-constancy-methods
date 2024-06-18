import numpy as np
import math
from scipy import ndimage
from .gray_pixels_yang import derivGauss

def gray_pixels_qian(img, mask = None, Npre = 0.01, delta_threshold=1e-4):
    '''
    img: 待处理的图片；
    mask: 图片掩模,将饱和像素和色卡标记为false，用于光源估计的像素标记为true
    num_gps: 所使用的会像素占比
    delta_threshold:对比度阈值
    '''
    num_GPS = math.floor(img.shape[0]*img.shape[1]*Npre/100) # 用于光照估计的灰像素数量
    eps = 1e-10 # 一个极小数，小于它就表示0

    # 当mask为none时，设定一个所有像素都被标记为True的掩模
    if mask is None:
        mask = np.ones((img.shape[0], img.shape[1]))==1
        
    mask = ~mask

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    # 去噪
    R = ndimage.uniform_filter(R, [7, 7], mode='wrap')
    G = ndimage.uniform_filter(G, [7, 7], mode='wrap')
    B = ndimage.uniform_filter(B, [7, 7], mode='wrap')
    # 消0
    mask = mask | (R == 0) | (G == 0) | (B == 0)
    R = np.where(R == 0, eps, R)
    G = np.where(G == 0, eps, G)
    B = np.where(B == 0, eps, B)
    norm1 = R + G + B

    # 标记出低对比度像素
    delta_R = derivGauss(R, 0.5)
    delta_G = derivGauss(G, 0.5)
    delta_B = derivGauss(B, 0.5)
    mask_low_contrast = (delta_R<=delta_threshold) & (delta_G<=delta_threshold) & (delta_B<=delta_threshold)
    mask = mask | mask_low_contrast

    # 取对数，并在对数空间求对比度
    log_R = np.log(R) - np.log(norm1)
    log_B = np.log(B) - np.log(norm1)
    delta_log_R = derivGauss(log_R, 0.5)
    delta_log_B = derivGauss(log_B, 0.5)
    mask = mask | np.isnan(delta_log_R) | np.isnan(delta_log_B)

    # 求灰指标
    data = np.zeros((delta_log_R.size, 2))
    data[:, 0] = delta_log_R.flatten('F')
    data[:, 1] = delta_log_B.flatten('F')
    mink_norm = 2
    norm2_data = np.power(np.sum(np.power(data, mink_norm), 1), 1/mink_norm)
    map_uniquelight = np.reshape(norm2_data, delta_log_R.shape, 'F')

    # 去除不要的像素点
    map_uniquelight[mask] = np.max(map_uniquelight)

    # 去噪
    map_uniquelight = ndimage.uniform_filter(map_uniquelight, [7, 7], mode='wrap')

    # 根据灰索引图来筛选像素
    Greyidx_unique = map_uniquelight
    sort_unique = np.sort(Greyidx_unique, axis=None)
    Gidx_unique = np.zeros_like(Greyidx_unique)
    Gidx_unique[Greyidx_unique<=sort_unique[num_GPS-1]] = 1

    # 计算筛选出的像素的平均值
    Gidx_unique = Gidx_unique[:, :, np.newaxis]
    illu_est = np.mean(img * Gidx_unique, axis=(0, 1))
    illu_est = illu_est/np.linalg.norm(illu_est, ord=2)
    
    return illu_est