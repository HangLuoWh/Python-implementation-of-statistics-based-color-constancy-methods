import numpy as np
from .gray_edge import set_boarder, norm_derivative,gDer
from math import sqrt
import math
import sys

def angle_error(est, gt):
    est_len = np.sum(est**2)**0.5
    gt_len = np.sum(gt**2)**0.5
    dot_product = np.sum(est*gt)
    angle = math.acos(dot_product/(est_len*gt_len + 1e-10 ))/math.pi*180
    return angle

def compute_edges(channel, sigma):
    x = gDer(channel, sigma, 1, 0)
    y = gDer(channel, sigma, 0, 1)
    return x, y, np.sqrt(x**2 + y**2)

def compute_spvar(img, sigma):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    Rx, Ry, Rw = compute_edges(R, sigma)
    Gx, Gy, Gw = compute_edges(G, sigma)
    Bx, By, Bw = compute_edges(B, sigma)

    # opponent_der
    x_sqrt = (Rx + Gx + Bx) / sqrt(3)
    y_sqrt = (Ry + Gy + By) / sqrt(3)

    sp_var = np.sqrt(x_sqrt**2 + y_sqrt**2)
    return sp_var, Rw, Gw, Bw

def weighted_gray_edge(img, img_mask, n=1, sigma=2, p=5, kappa=10):
    '''
    img: 图像
    img_mask: 图像掩模,将饱和像素和色卡标记为0，用于光源估计的像素标记为1
    n: 求导的阶数, 0~2
    sigma: 高斯核标准差，非负
    p: p范数（-1到正无穷）
    kappa: 用于强调权重的参数
    注意：与原始matlab实现不同，本实现并未对img_mask进行上下左右的膨胀，这一简化对结果影响不大。
    '''
    iter = 100 # 迭代次数
    eps = sys.float_info.epsilon # 最小正浮点数，小于它的值直接视为0
    # 初始估计,设定为白光
    final_ill = np.array([1/sqrt(3), 1/sqrt(3), 1/sqrt(3)])
    tmp_ill = np.array([1/sqrt(3), 1/sqrt(3), 1/sqrt(3)])# 存放中间光源估计结果的临时变量

    flag = 1
    while(iter > 0 and flag == 1):
        iter -= 1 # 迭代次数

        img[:, :, 0] = img[:, :, 0]/(sqrt(3)*tmp_ill[0])
        img[:, :, 1] = img[:, :, 1]/(sqrt(3)*tmp_ill[1])
        img[:, :, 2] = img[:, :, 2]/(sqrt(3)*tmp_ill[2])

        sp_var, Rw, Gw, Bw = compute_spvar(img, sigma)

        mask_zero_edge = (Rw < eps) & (Gw < eps) & (Bw < eps)
        mask_zero_edge = ~mask_zero_edge
        mask = img_mask & mask_zero_edge
        mask = set_boarder(mask, sigma+1)

        grad_im = np.sqrt(Rw**2 + Gw**2 + Bw**2)
        weight_map = (sp_var/grad_im)**kappa
        weight_map[weight_map>1] = 1

        data_Rx = np.power(Rw*weight_map, p)
        data_Gx = np.power(Gw*weight_map, p)
        data_Bx = np.power(Bw*weight_map, p)

        tmp_ill[0] = np.power(np.sum(data_Rx[mask]), 1/p)
        tmp_ill[1] = np.power(np.sum(data_Gx[mask]), 1/p)
        tmp_ill[2] = np.power(np.sum(data_Bx[mask]), 1/p)
        tmp_ill = tmp_ill/np.linalg.norm(tmp_ill, 2)

        final_ill = final_ill*tmp_ill
        final_ill = final_ill/np.linalg.norm(final_ill, 2)
        
        if (angle_error(tmp_ill, np.array([1/sqrt(3), 1/sqrt(3), 1/sqrt(3)])) < 0.05):
            flag = 0
            
    return final_ill