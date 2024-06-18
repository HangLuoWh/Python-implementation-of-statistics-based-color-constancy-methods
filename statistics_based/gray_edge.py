import numpy as np
import math
import cv2

def set_boarder(mask, width):
    '''
    将掩模的边缘width行和列设置为0
    mask：掩模
    width：拓展宽度
    method：可选值为0或1，0代表0填充，1表示平均填充
    '''
    tmp = np.ones_like(mask)
    xx, yy = np.meshgrid(range(tmp.shape[0]), range(tmp.shape[1]))
    xx = xx.transpose() # xx为行索引
    yy = yy.transpose() # yy为列索引
    tmp = tmp*((yy < tmp.shape[1] - width) & (yy > width-1)) # 将左右width列设置为0
    tmp = tmp*((xx < tmp.shape[0] - width) & (xx > width-1)) # 将上下width行设置为0
    # 将mask中的边缘行列设置为0
    out = mask*tmp
    return out

def fill_boarder(img, bw):
    '''
    对图像边缘以复制的方式进行填充
    img: 待填充的图像
    bw: 填充大小
    '''
    hh = img.shape[0]
    ww = img.shape[1]
    dimension = img.ndim

    if (dimension == 2):
        out = np.zeros((hh + bw*2, ww + bw*2))
        # 填充四个角点的值
        out[0:bw, 0:bw] = np.ones((bw, bw))*img[0, 0]
        out[0:bw, ww+bw:ww+2*bw] = np.ones((bw, bw))*img[0, ww-1]
        out[hh+bw:hh+2*bw, 0:bw] = np.ones((bw, bw))*img[hh-1, 0]
        out[hh+bw:hh+2*bw, ww+bw:ww+2*bw] = np.ones((bw, bw))*img[hh-1, ww-1]
        # 填充中间位置
        out[bw:hh+bw,bw:ww+bw] = img
        # 复制填充四个边缘(上下左右)
        out[0:bw, bw:ww+bw] = np.ones((bw, ww))*img[0,:]
        out[bw+hh:2*bw+hh, bw:ww+bw] = np.ones((bw, ww))*img[-1,:]
        out[bw:hh+bw, :bw] = np.ones((hh, bw))*img[:, :1]
        out[bw:hh+bw, bw+ww:2*bw+ww] = np.ones((hh, bw))*img[:, -1:] 
    else:
        out = np.zeros((hh + bw*2, ww + bw*2, dimension))
        for i in range(dimension):
            # 填充四个角点的值
            out[0:bw, 0:bw, i] = np.ones((bw, bw))*img[0, 0, i]
            out[0:bw, ww+bw:ww+2*bw, i] = np.ones((bw, bw))*img[0, ww-1, i]
            out[hh+bw:hh+2*bw, 0:bw, i] = np.ones((bw, bw))*img[hh-1, 0, i]
            out[hh+bw:hh+2*bw, ww+bw:ww+2*bw, i] = np.ones((bw, bw))*img[hh-1, ww-1, i]
            # 填充中间位置
            out[bw:hh+bw,bw:ww+bw, i] = img[:, :, i]
            # 复制填充四个边缘(上下左右)
            out[0:bw, bw:ww+bw, i] = np.ones((bw, ww))*img[0,:,i]
            out[bw+hh:2*bw+hh, bw:ww+bw, i] = np.ones((bw, ww))*img[-1,:,i]
            out[bw:hh+bw, :bw, i] = np.ones((hh, bw))*img[:, :1, i]
            out[bw:hh+bw, bw+ww:2*bw+ww, i] = np.ones((hh, bw))*img[:, -1:, i]

    return out

def gDer(f, sigma, iorder, jorder):
    '''
    求图像梯度
    sigma: 算子的半宽度
    iorder: 行方向上的阶数
    jorder: 列方向上的阶数
    '''
    break_off_sigma = 3.
    filtersize = math.floor(break_off_sigma*sigma + 0.5)
    f = fill_boarder(f, filtersize) # 复制填充图像边缘
    x = np.arange(-filtersize, filtersize+1, 1) # 高斯函数的坐标
    gauss = 1/(np.sqrt(2*math.pi)*sigma)*np.exp((x**2)/(-2*sigma**2)) # 高斯函数
    
    # 横向滤波
    if(iorder == 0):
        # 高斯平滑
        Gx = gauss / np.sum(gauss)
    elif(iorder == 1):
        # 一维高斯函数的导数
        Gx = -(x/sigma**2)*gauss
        Gx = Gx/(np.sum(x*Gx))
    elif(iorder == 2):
        # 二维高斯函数的导数
        Gx = (x**2/sigma**4 - 1/sigma**2)*gauss
        Gx = Gx - np.sum(Gx)/x.size
        Gx = Gx/np.sum(0.5*x*x*Gx)
    Gx = Gx.reshape(1, -1) # 将列向量转为行向量
    H = cv2.filter2D(f, -1, Gx, borderType=cv2.BORDER_CONSTANT) # 0填充边缘

    # 纵向滤波
    if(jorder == 0):
        # 高斯平滑
        Gy = gauss/np.sum(gauss)
    elif(jorder == 1):
        # 一维高斯函数的导数
        Gy = -(x/sigma**2)*gauss
        Gy = Gy/(np.sum(x*Gy))
    elif(jorder == 2):
        # 二维高斯函数的导数
        Gy = (x**2/sigma**4 - 1/sigma**2)*gauss
        Gy = Gy - np.sum(Gy)/x.size
        Gy = Gy/np.sum(0.5*x*x*Gy)
    Gy = Gy.reshape(-1, 1)
    H = cv2.filter2D(H, -1, Gy, borderType=cv2.BORDER_CONSTANT) # 0填充边缘

    return H[filtersize:H.shape[0]-filtersize, filtersize:H.shape[1]-filtersize]

def norm_derivative(img, sigma, order, min_norm):
    '''
    用高斯梯度算子对图像滤波
    img：待处理的图像；
    sigma：高斯梯度算子的宽度；
    order：高斯梯度算子的阶数；
    min_norm：p范数的阶数。
    '''
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    if order == 0:
        if sigma != 0:
            Rw = gDer(R, sigma, 0, 0)
            Gw = gDer(G, sigma, 0, 0)
            Bw = gDer(B, sigma, 0, 0)
        else:
            Rw = R
            Gw = G
            Bw = B
    elif order == 1: # 1阶梯度
        Rx = gDer(R, sigma, 1, 0)
        Ry = gDer(R, sigma, 0, 1)
        Rw = np.sqrt(Rx**2 + Ry**2)

        Gx = gDer(G, sigma, 1, 0)
        Gy = gDer(G, sigma, 0, 1)
        Gw = np.sqrt(Gx**2 + Gy**2)

        Bx = gDer(B, sigma, 1, 0)
        By = gDer(B, sigma, 0, 1)
        Bw = np.sqrt(Bx**2 + By**2)
    elif order == 2: # 2阶梯度
        Rxx = gDer(R, sigma, 2, 0)
        Ryy = gDer(R, sigma, 0, 2)
        Rxy = gDer(R, sigma, 1, 1)
        Rw = np.sqrt(Rxx**2 + 4*Rxy**2 + Ryy**2)

        Gxx = gDer(G, sigma, 2, 0)
        Gyy = gDer(G, sigma, 0, 2)
        Gxy = gDer(G, sigma, 1, 1)
        Gw = np.sqrt(Gxx**2 + 4*Gxy**2 + Gyy**2)

        Bxx = gDer(B, sigma, 2, 0)
        Byy = gDer(B, sigma, 0, 2)
        Bxy = gDer(B, sigma, 1, 1)
        Bw = np.sqrt(Bxx**2 + 4*Bxy**2 + Byy**2)
    else:
        raise NotImplementedError('阶数超出范围（0~2）')
    
    return Rw, Gw, Bw


def gray_edge(img, img_mask, n, sigma, p):
    '''
    img: 图像
    img_mask: 图像掩模,将饱和像素和色卡标记为0，用于光源估计的像素标记为1
    n: 求导的阶数, 0~2
    sigma: 高斯核标准差，非负
    p: p范数（-1到正无穷）
    注意：与原始matlab实现不同，本实现并未对img_mask进行上下左右的膨胀，这一简化对结果影响不大。
    '''
    img_mask2 = set_boarder(img_mask, 3)

    # 求图像梯度
    Rx, Gx, Bx = norm_derivative(img, sigma, n, p)
    Rx = np.abs(Rx)
    Gx = np.abs(Gx)
    Bx = np.abs(Bx)
    
    # 计算梯度图的p范数
    if p != -1:
        Rx_p = np.power(Rx, p)
        Gx_p = np.power(Gx, p)
        Bx_p = np.power(Bx, p)

        white_R = np.power(np.sum(Rx_p*img_mask2), 1/p)
        white_G = np.power(np.sum(Gx_p*img_mask2), 1/p)
        white_B = np.power(np.sum(Bx_p*img_mask2), 1/p)
    else: # 若为-1，则求最大值
        white_R = np.max(Rx*img_mask2)
        white_G = np.max(Gx*img_mask2)
        white_B = np.max(Bx*img_mask2)

    white_len = np.sqrt(white_R**2+white_G**2+white_B**2)

    white_R = white_R/white_len
    white_G = white_G/white_len
    white_B = white_B/white_len

    return np.array([white_R, white_G, white_B])