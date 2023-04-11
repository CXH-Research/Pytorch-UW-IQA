from PIL import Image
import torch
import torch.nn.functional as F
import os
from torchvision.transforms.functional import to_tensor
import math

def _uiconm(x, window_size):
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[2]/window_size)
    k2 = int(x.shape[1]/window_size)
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:, 0:blocksize_y*k2, 0:blocksize_x*k1]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[:,k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            if block.shape[1] == 0 or block.shape[2] == 0:
                max_ = 0
                min_ = 0
            else:
                max_ = torch.max(block)
                min_ = torch.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),torch.log(top/bot))
    return w*val

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = x.sort()[0]
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = torch.sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = torch.sum(torch.pow(x - mu, 2)) / len(x)
    return val


def _uicm(x):
    R = x[0,:,:].flatten()
    G = x[1,:,:].flatten()
    B = x[2,:,:].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = torch.sqrt( (torch.pow(mu_a_RG,2)+torch.pow(mu_a_YB,2)) )
    r = torch.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[0,:,:]
    G = x[1,:,:]
    B = x[2,:,:]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel_torch(R)
    Gs = sobel_torch(G)
    Bs = sobel_torch(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = torch.multiply(Rs, R)
    G_edge_map = torch.multiply(Gs, G)
    B_edge_map = torch.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = torch.max(block)
            min_ = torch.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += torch.log(max_/min_)
    return w*val

def sobel_torch(x):
    x = x.squeeze(0)
    dx = F.conv2d(x[None, None], sobel_kernel_x.to(x.device), padding=1)
    dy = F.conv2d(x[None, None], sobel_kernel_y.to(x.device), padding=1)
    mag = torch.hypot(dx, dy)
    mag *= 255.0 / torch.max(mag)
    return mag.squeeze()

sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

# Replace all instances of torch with torch and ndimage.sobel with sobel_torch in the original code
# Also, convert the image to tensor at the beginning and back to NumPy array at the end of the function

def getUIQM(x):
    x = Image.open(x)
    x = to_tensor(x) * 255
    x = x.cuda()
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm.item()