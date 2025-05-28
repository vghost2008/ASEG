import torch
import torch.nn as nn
import glob
import os
import os.path as osp
from functools import partial
import argparse
from utils import setup_seed
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import tifffile
import cv2
import copy
import numpy as np

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
# from dataset import get_strong_transforms
from dataset3 import get_strong_transforms_sig
from utils import get_gaussian_kernel, cal_anomaly_maps

def set_model(args):
    setup_seed(1)
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    
    print(f'use {args.pretrained_model_path}/{args.item} best model')
    model.load_state_dict(torch.load(osp.join(args.pretrained_model_path, args.item, 'best_model.pth'), map_location='cpu'))

    return model

def read_best_thr(args) -> dict:
    thrs = {}
    record_path = osp.join(args.pretrained_model_path, 'record.txt')
    if osp.exists(record_path):
        with open(record_path) as f:
            lines = f.readlines()
            for line in lines:
                line_sps = line.split(':')
                item_name = line_sps[0]
                thr = float(line_sps[-1].replace(' ', ''))
                thrs[item_name] = thr
    else:
        raise FileExistsError('record.txt is not exist')
    return thrs
    

def get_image_path(data_root):
    img_paths = glob.glob(data_root + "/*.png") + \
                            glob.glob(data_root + "/*.JPG") + \
                            glob.glob(data_root + "/*.bmp")
    return img_paths

def save_tiff(anomaly_map, save_root, img_path, thresholded=None):
    batch_num = anomaly_map.shape[0]
    for i in range(batch_num):
        img_path_list = img_path[i].split('/')
        class_name, phase, img_name = img_path_list[-3], img_path_list[-2], img_path_list[-1]  
        img_name = os.path.splitext(img_name)[0] + '.tiff'

        anomaly_map_np = copy.deepcopy(anomaly_map[i].squeeze(0).cpu().detach().numpy().astype(np.float16)) # anomaly_map.cpu().numpy()

        save_path = os.path.join(save_root, "anomaly_images", class_name, phase, img_name)  
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tifffile.imwrite(save_path, anomaly_map_np)
        if thresholded is not None and isinstance(thresholded, float):
            anomaly_map_np[anomaly_map_np >= thresholded] = 255
            anomaly_map_np[anomaly_map_np < thresholded] = 0
            save_path = os.path.join(save_root, "anomaly_images_thresholded", class_name, phase, img_name.replace('tiff', 'png'))  
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            anomaly_map_np = anomaly_map_np.astype(np.uint8)
            cv2.imwrite(save_path, anomaly_map_np)


def pred_images(args, model, thr:float=None):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    img_paths = get_image_path(osp.join(args.data_path, args.item, "test_private"))
    img_paths.extend(get_image_path(osp.join(args.data_path, args.item, "test_private_mixed")))

    _, test_data_transform, _ = get_strong_transforms_sig(args.input_size, args.input_size)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img_path in tqdm(img_paths, ncols=80):
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
            img:torch.Tensor = test_data_transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)
            save_dir = args.save_dir + "_tiff" + "/predict"
            save_tiff(anomaly_map, save_dir, [img_path,], thresholded=thr)

    
if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=784)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--save_dir', type=str, default='./work-dir')
    parser.add_argument('--pretrained_model_path', type=str, default="inpformer2")
    parser.add_argument('--data_path', type=str, default='/path/to/MVTecAD2')
    parser.add_argument('--item', type=str, default='can')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    args.pretrained_model_path = osp.join(args.save_dir, args.pretrained_model_path)

    item_thrs = read_best_thr(args)
    print(item_thrs)
    if args.item in item_thrs:
        thr = item_thrs[args.item]
        print(f'========= process {args.item}: {thr}')
        torch_model = set_model(args)
        pred_images(args, torch_model, thr=thr)
        del torch_model
        