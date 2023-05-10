import cv2
import math
import numpy as np
import kornia.color as color
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch


def uciqe(image):
    image = cv2.imread(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    # 求亮度对比值
    n, m = np.shape(V)
    number = math.floor(n * m / 100)
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    v = -v
    v.sort()
    v = -v
    top = np.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe

# improve the following code to support batch operation:

def torch_uciqe(image):
    # RGB转为HSV
    hsv = color.rgb_to_hsv(image)  
    H, S, V = torch.chunk(hsv, 3)

    # 色度的标准差
    delta = torch.std(H) / (2 * math.pi)
    
    # 饱和度的平均值
    mu = torch.mean(S)  
    
    # 求亮度对比值
    n, m = V.shape[1], V.shape[2]
    number = math.floor(n * m / 100)
    v = V.flatten()
    v, _ = v.sort()
    bottom = torch.sum(v[:number]) / number
    v = -v
    v, _ = v.sort()
    v = -v
    top = torch.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe

if __name__ == '__main__':
    image = 'test.png'
    img = Image.open(image).convert('RGB')
    img = to_tensor(img).cuda().unsqueeze(0)
    img = torch.cat((img, img), 0)
    print(img.shape)
    res = torch.stack([torch_uciqe(res) for res in img], dim=0)
    print(res.shape)
    print(res)
    