import torch
import torch.nn as nn
import numpy as np
import os
import copy
import shutil
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

# Dataset-Related Modules
# from dataset import MVTecDataset, RealIADDataset, MVTec2Dataset
# from dataset import get_data_transforms, get_strong_transforms
# from dataset2 import get_strong_transforms, get_strong_transforms_without_norm, get_strong_transforms_sig, MVTec2DatasetV2, MVTec2DatasetV3, MVTec2DatasetV2Fast
import dataset3
from dataset3 import get_strong_transforms_sig, MVTec2DatasetV2Fast
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models import vit_encoder
from models import uad
from models.uad import INP_Former, MIX_Former2, MIX_Former3, MultiBranchStemF2X
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block


warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)
    # Data Preparation
    if args.dataset == 'MVTec-AD2':
        train_data_transforms, test_data_transforms, gt_transforms = get_strong_transforms_sig(args.input_size, args.crop_size)
        # if args.item in ['can', 'fruit_jelly', 'vial']:
        #     train_data_transforms = train_data_transforms2
        data_root = os.path.join(args.data_path, args.item)
        train_data = MVTec2DatasetV2Fast(root=data_root, transform=train_data_transforms, gt_transform=gt_transforms, phase='train')
        val_data = MVTec2DatasetV2Fast(root=data_root, transform=test_data_transforms, gt_transform=gt_transforms, phase='val')

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        raise NotImplementedError('Only support MVTec-AD2')

    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # stem
    # s2_stem = MultiBranchStemF2X(3, 3)
    # Encoder info
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
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-5, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train {} image number:{}'.format(args.item, len(train_data)))

        loss_weights = [2.0, 0.2]

        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        if args.resume and os.path.exists(os.path.join(args.save_dir, args.save_name, args.item, 'best_model.pth')):
            model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'best_model.pth')), strict=True)
        # Train
        min_loss = None
        min_loss_ep = 0
        for epoch in range(args.total_epochs):
            model.train()
            cos_losses = []
            g_losses = []
            loss_list = []
            for data in tqdm(train_dataloader, ncols=80):
                img = data[0]
                img = img.to(device)
                en, de, g_loss = model(img)
                cos_loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss_weights[0] * cos_loss + loss_weights[1] * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()
                cos_losses.append(cos_loss.item() * loss_weights[0])
                g_losses.append(g_loss.item() * loss_weights[1])
                loss_list.append(loss.item())
            print_fn('{} epoch [{}/{}], loss:{:.4f}, cos_loss:{:.4f}, g_loss:{:.4f}'.format(args.item, epoch+1, args.total_epochs, 
                                                                                                          np.mean(loss_list),
                                                                                                          np.mean(cos_losses),
                                                                                                          np.mean(g_losses)))
            # 保留一个最新的模型和loss最小的模型
            if min_loss is None:
                min_loss = np.mean(loss_list).item()
                min_loss_ep = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, f'best_model.pth'))
            else:
                if min_loss >= np.mean(loss_list).item():
                    min_loss = np.mean(loss_list).item()
                    min_loss_ep = epoch + 1
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'best_model.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'latest_model.pth'))
            # if (epoch + 1) % args.total_epochs == 0:
        
        # 测试loss最小的模型
        print_fn(f'the best min loss: {min_loss:.4f} in epoch {min_loss_ep}')
        best_model = model
        best_model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'best_model.pth')), strict=True)
        best_model.eval()
        results = evaluation_batch(best_model, test_dataloader, device, max_ratio=0.01, resize_mask=784)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        seg_f1 = f1_px if not isinstance(f1_px, tuple) else f1_px[0]
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, seg_f1, aupro_px))
        record_save_path = os.path.join(args.save_dir, args.save_name, 'record.txt')
        with open(record_save_path, mode='a+', encoding='utf-8') as f:
            f.write(f'{args.item}: seg-f1 score: {f1_px[0]}, thr: {f1_px[1]}\n')
        return results
    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'best_model.pth')), strict=True)
        model.eval()
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=784)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        return results



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD2') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'/home/zhangym/workspace-27/datasets/AnomalyDetectionDatasets/MVTecAD2')  # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./work-dir')
    parser.add_argument('--save_name', type=str, default='TUAD-Single-Class-v2')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--crop_size', type=int, default=784)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--phase', type=str, default='train')
    # parser.add_argument(
    #     '--device', nargs='+', type=str, default='cpu', help='cuda device')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume train')

    args = parser.parse_args()
    # args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    resume = args.resume

    # category info
    if args.dataset == 'MVTec-AD':
        # args.data_path = 'E:\IMSN-LW\dataset\mvtec_anomaly_detection' # '/path/to/dataset/MVTec-AD/'
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'MVTec-AD2':
        args.item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        # args.data_path = 'E:\IMSN-LW\dataset\Real-IAD'  # '/path/to/dataset/Real-IAD/'
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']

    # if resume:
    #     saved_items = os.listdir(os.path.join(args.save_dir, args.save_name))
    #     args.item_list = sorted(list(set(args.item_list).difference(set(saved_items))))

    # args.item_list = args.item_list[6:]
    print_fn(f'cfg: {__file__}\nargs: {args}')
    # backup cfg files
    shutil.copy2(__file__, os.path.join(args.save_dir, args.save_name))
    shutil.copy2(dataset3.__file__, os.path.join(args.save_dir, args.save_name))
    shutil.copy2(uad.__file__, os.path.join(args.save_dir, args.save_name))

    # result_list = []
    for item in args.item_list:
        args.item = item
        main(args)
        # auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = main(args)
        # result_list.append([args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

    # mean_auroc_sp = np.mean([result[1] for result in result_list])
    # mean_ap_sp = np.mean([result[2] for result in result_list])
    # mean_f1_sp = np.mean([result[3] for result in result_list])

    # mean_auroc_px = np.mean([result[4] for result in result_list])
    # mean_ap_px = np.mean([result[5] for result in result_list])
    # mean_f1_px = np.mean([result[6] for result in result_list])
    # mean_aupro_px = np.mean([result[7] for result in result_list])

    # print_fn(result_list)
    # print_fn(
    #     'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
    #         mean_auroc_sp, mean_ap_sp, mean_f1_sp,
    #         mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
