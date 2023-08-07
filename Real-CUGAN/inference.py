
import argparse
import torch
from torch import nn as nn
from torch.nn import functional as F
import os
import cv2
import numpy as np
from upcunet_v3 import RealWaifuUpScaler

upscalers = {}
def process_image(image, scale=2, tile=5, save=False, save_dir = 'finish/result.png'):
    global upscalers
    torch.cuda.empty_cache()
    result = upscalers[scale](image,tile_mode=tile,cache_mode=2,alpha=1)
    if save:
        save_result(result, save_dir)
    return result

def save_result(image, filepath = 'result.png'):
    cv2.imwrite(filepath, image[:, :, [2, 1, 0]])
    
def process_image_path(path, scale=2, tile=5, save=False, save_dir = 'finish/result.png'):
    img = cv2.imread(path)[:, :, [2, 1, 0]]
    result = process_image(img, scale, tile)
    if save:
        save_result(result, save_dir)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name", default="up2x-latest-no-denoise.pth")
    parser.add_argument("--tile", help="tile size", default=4)
    parser.add_argument("--model_path", help="model path", default="model_weights/")
    parser.add_argument("--finish_path", help="finish path", default="finish/")
    parser.add_argument("--image-path", help="finish path", default="test.png") # add more arguments by need
    args = parser.parse_args()

    ROOTPATH="." # root dir (a constant)
    ModelPath=os.path.abspath(os.path.join(ROOTPATH, args.model_path, args.model)) if not os.path.isabs(args.model_path) else args.model_path # model dir
    FinishPath=os.path.abspath(os.path.join(ROOTPATH, args.finish_path)) if not os.path.isabs(args.finish_path) else args.finish_path # output dir
    ModelName=args.model # default model
    Tile=args.tile # default tile size
    image_target=args.image_path # image path 
    assert Tile in {0,1,2,3,4}, "Tile size must be 0,1,2,3,4"

    # check model path exists. for pending path and finish path, make directory if not exists
    assert os.path.exists(ModelPath), "Model path does not exist"
    #print(FinishPath)
    if not os.path.exists(FinishPath):
        os.mkdir(FinishPath)
    upscalers[2] = RealWaifuUpScaler(2, ModelPath, half=True, device="cuda:0") # setup upscalers
    process_image_path(image_target, scale=2, tile=Tile, save=True, save_dir = os.path.join(FinishPath,"result.png")) # call this to inference

    



