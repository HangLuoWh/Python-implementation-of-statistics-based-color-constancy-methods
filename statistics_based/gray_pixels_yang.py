import numpy as np
import math
from scipy import ndimage

def localStd(channel, wsize):
    '''
    局部标准差滤波
    channel: 取过对数的单个图像通道
    wsize：局部范围
    '''

    half_height = math.floor(wsize[0]/2)
    half_width = math.floor(wsize[1]/2)

    # 计算局部标准差
    emap = np.pad(channel, [half_height, half_width], mode = 'edge') # 边缘填充

    # 局部标注差求法1（更快）
    emap_square = ndimage.uniform_filter(emap**2, wsize, mode='constant')
    emap_uniform = ndimage.uniform_filter(emap, wsize, mode='constant')
    difference = emap_square - emap_uniform**2
    difference = np.where(difference<0, 0, difference)
    contrast = np.sqrt(difference)

    # # 局部标注差求法2（慢很多）
    # contrast = ndimage.generic_filter(emap, np.std, wsize, mode='constant')

    contrast = contrast*np.sqrt((wsize[0]*wsize[1])/(wsize[0]*wsize[1]-1)) # 将总体标注差（numpy的计算方式）转为样本标准差（matlab的计算方式）
    # 均匀的位置在我这边算出来为0，在matlab算出来为一个极小数
    ex = contrast.shape[0]
    ey = contrast.shape[1]

    return contrast[half_height:ex-half_height, half_width:ey-half_width]

def derivGauss(channel, sigma):
    '''
    高斯梯度滤波
    channel: 对数运算后的单个图像通道
    sigma：滤波器半径
    '''

    gaussian_die_off = 0.000001 # 高斯分布的截断阈值
    pw = np.arange(1, 51, 1)
    ssq = sigma**2

    # 只考虑高斯函数的值大于gaussian_die_off的范围
    width = np.where(np.exp(-pw**2/(2*ssq)) > gaussian_die_off)
    if width[0].size == 0:
        width = 1
    else:
        width = width[0][-1] + 1
    
    # 构造高斯梯度核
    x, y = np.meshgrid(np.arange(-width, width+1, 1), np.arange(-width, width+1, 1)) # 生成网格坐标
    dgau2D = -x*np.exp(-(x**2+y**2)/(2*ssq))/(math.pi*ssq)

    # 真卷积
    ax = ndimage.filters.convolve(channel, dgau2D, mode='nearest')
    ay = ndimage.filters.convolve(channel, np.transpose(dgau2D), mode='nearest')

    return np.sqrt(ax**2 + ay**2)
    
def gray_idx(img, scale, method):
    '''
    灰度指标计算
    img:输入图像
    scale:滤波器尺寸
    method：edge为边缘，std为标注差
    '''
    eps = 1e-10 # 一个极小数，小于它就表示0
    rr = img.shape[0] # 高
    cc = img.shape[1] # 宽

    # 分通道，排除0，为对数运算作准备
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    R = np.where(R == 0, eps, R)
    G = np.where(G == 0, eps, G)
    B = np.where(B == 0, eps, B)

    if method == 'edge': # 使用边缘计算灰系数
        dr = derivGauss(np.log(R), scale)
        dg = derivGauss(np.log(G), scale)
        db = derivGauss(np.log(B), scale)
    elif method == 'std': # 使用标注差计算灰系数
        wsize = [scale, scale]
        dr = localStd(np.log(R), wsize)
        dg = localStd(np.log(G), wsize)
        db = localStd(np.log(B), wsize)
    else:
        raise NotImplementedError(f"{method}未实现，请使用'edge'或'std'方法")
    
    # 为避免舍入误差，将dr、dg、db中小于eps的值直接归0
    dr = np.where(dr < eps, 0, dr)
    dg = np.where(dg < eps, 0, dg)
    db = np.where(db < eps, 0, db)

    # 求相对标准差P,作为光照不变指标，越小表示越接近于灰
    data = np.array([dr.flatten(), dg.flatten(), db.flatten()])
    data_std = np.std(data, axis=0)*np.sqrt(3/2) # 将总体标注差（numpy的计算方式）转为样本标准差（matlab的计算方式）
    data_std = data_std/(np.mean(data, 0) + eps)

    # 相对标准差除以每个像素的亮度得到GI，以削弱暗像素对灰指标的影响
    data1 = np.array([R.flatten(), G.flatten(), B.flatten()])
    ps = data_std/(np.mean(data1, axis=0) + eps)
    gray_idx = np.reshape(ps, [rr, cc])
    gray_idx = gray_idx/(np.max(gray_idx) + eps)

    # 排除三个通道对比度均为0的像素
    gray_idx = np.where((dr == 0) & (dg == 0) & (db == 0), np.max(gray_idx), gray_idx)

    gray_idx = ndimage.uniform_filter(gray_idx, [7, 7], mode='wrap')
    gray_idx = np.where(gray_idx < 0, 0, gray_idx) # 因为舍入误差的存在，均匀滤波会引入负值，此处直接将负值归0
    return gray_idx

def gray_pixels(img, method = 'std', mask = None, Npre = 0.01):
    '''
    img: 待处理的图片；
    method：用于计算灰度指标的方法，edge为边缘，std为标注差;
    mask: 图片掩模,将饱和像素和色卡标记为0，用于光源估计的像素标记为1
    num_gps: 所使用的会像素占比
    '''
    num_GPS = math.floor(img.shape[0]*img.shape[1]*Npre/100) # 用于光照估计的灰像素数量
    
    if method == 'edge':
        sigma  = 0.5 # 论文默认参数
        grayidx = gray_idx(img,sigma, method) # 求每个像素的灰度系数
    elif method == 'std':
        grayidx = gray_idx(img, 3, method) # 求每个像素的灰度系数，3为论文默认设置
    else:
        raise NotImplementedError(f"{method} 未实现，请使用'edge'或'std'方法")
    
    if mask is not None:
        grayidx[~mask] = np.max(grayidx)

    tt = np.sort(grayidx, axis=None)
    Gidx = np.zeros_like(grayidx)
    Gidx[grayidx <= tt[num_GPS-1]] = 1
    Gidx = Gidx[:, :, np.newaxis]

    illu_est = np.sum(img*Gidx, axis=(0, 1))
    return illu_est