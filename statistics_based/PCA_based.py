import numpy as np
import math

def PCA_based(img, per, mask=None):
    '''
    基于pca的方法
    img: 图像
    per: 亮/暗图像的占比
    mask: 图片掩模,将饱和像素和色卡标记为0，用于光源估计的像素标记为1
    '''
    maximum =  2**16-1
    img_reshape = np.reshape(img, (-1, 3), 'F')/maximum

    if mask is not None:
        img_mask = np.reshape(mask, -1, 'F')
        img_reshape = img_reshape[img_mask==1]

    n = img_reshape.shape[0] # 像素数量
    # gray world估计
    l = np.mean(img_reshape, axis=0)
    l = l/np.linalg.norm(l)
    # 像素在l上作映射,即点积，再进行排序
    img_prj = np.sum(img_reshape*l, 1)
    img_prj_idx = np.argsort(img_prj, axis=0)

    # 选取前per%和后per%的像素
    sel_idx = np.append(img_prj_idx[:math.ceil(n*per/100) + 1], img_prj_idx[math.floor(n*(100-per)/100):])
    pixel_sel = img_reshape[sel_idx, :]

    # 求特征值和特征向量
    sigma = pixel_sel.transpose() @ pixel_sel
    values, vectors = np.linalg.eig(sigma)
    eig_sort_idx = np.argsort(values)
    vectors = vectors[:, eig_sort_idx]

    return np.abs(vectors[:, -1])