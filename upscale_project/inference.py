import torch
from torch import nn as nn
from torch.nn import functional as F
import os
import cv2
import numpy as np
from upcunet_v3 import onoma_upscaler 




upscaler = RealWaifuUpScaler(2, ModelPath, half=True, device="cuda:0")


process_image_path(image_target, scale=2, tile=Tile, save=True, save_dir = os.path.join(FinishPath,"result.png")) # call this to inference

    
def process_image_path(path, scale=2, tile=5, save=False, save_dir = 'finish/result.png'):
    img = cv2.imread(path)[:, :, [2, 1, 0]]

    torch.cuda.empty_cache()
    result = upscaler(img, tile_mode=tile, cache_mode=2, alpha=1)

    if save:
        cv2.imwrite(save_dir, result[:, :, [2, 1, 0]])

    return result