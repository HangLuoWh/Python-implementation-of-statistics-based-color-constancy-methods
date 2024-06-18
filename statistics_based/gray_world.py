import numpy as np

def gray_world(img, mask = None):
    '''
    img: 待处理的图片；
    mask: 图片掩模,将饱和像素和色卡标记为0，用于光源估计的像素标记为1
    '''
    img_reshape = np.reshape(img, (-1, 3))

    # 抽取出用于光源估计的像素
    if mask is not None:
        mask = mask.ravel()
        img_reshape = img_reshape[mask==1, :]

    img_reshape = np.reshape(img, (-1, 3))
    ill_est = np.mean(img_reshape, axis = 0)

    return ill_est