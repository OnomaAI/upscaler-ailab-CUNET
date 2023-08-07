import torch
from torch import nn as nn
from torch.nn import functional as F
import os
import cv2
import numpy as np
from onoma_upscaler import RealWaifuUpScaler


ModelPath = '/Data2/spacewebui/ailab/Real-CUGAN/model_weights/up2x-latest-no-denoise.pth'

upscaler = RealWaifuUpScaler(2, ModelPath, half=True, device="cuda:0")


def process_image_path(path, scale=2, save=False, save_dir = 'finish/result.png'):
    img = cv2.imread(path)[:, :, [2, 1, 0]]

    torch.cuda.empty_cache()
    result = upscaler(img, tile_mode=5, cache_mode=2, alpha=1)

    if save:
        cv2.imwrite(save_dir, result[:, :, [2, 1, 0]])

    return result


process_image_path('./test.png', scale=2, save=True, save_dir = './result.png') # call this to inference