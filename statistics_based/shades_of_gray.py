import numpy as np

def shades_of_gray(img, p, mask = None):
    '''
    img: 待处理的图片；
    p: 范数的阶数
    mask: 图片掩模,将饱和像素和色卡标记为0，用于光源估计的像素标记为1
    '''
    img_reshape = np.reshape(img, (-1, 3))

    # 抽取出用于光源估计的像素
    if mask is not None:
        mask = mask.ravel()
        img_reshape = img_reshape[mask==1, :]

    img_p = np.mean(np.power(img_reshape, p), axis = 0)
    illu_est = np.power(img_p, 1/p)
    return illu_est