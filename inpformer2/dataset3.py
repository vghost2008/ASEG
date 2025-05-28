import random

from torchvision import transforms
# from seg_transforms import seg_transforms
from PIL import Image
import os
import os.path as osp
import torch
import glob
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')

def get_strong_transforms_sig(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    list_aug = [
        transforms.ColorJitter(contrast=0.4),
        transforms.ColorJitter(brightness=0.4),
        transforms.ColorJitter(saturation=0.4, hue=0.1),
        transforms.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2)),
    ]
    aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

    transform_aug = [
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([ list_aug[i] for i in aug_idx], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
    ]

    train_data_transforms = transforms.Compose(transform_aug)

    test_data_transforms = transforms.Compose([
        transforms.Resize((isize, isize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
        ])
    gt_transforms = transforms.Compose([
        transforms.Resize((isize, isize)),
        transforms.ToTensor()
        ])
    return train_data_transforms, test_data_transforms, gt_transforms


class MVTec2DatasetV2Fast(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        assert phase in ('train', 'test', 'val')
        self.data_path = []
        self.phase = phase

        if phase == 'train':
            self.data_path = [
                os.path.join(root, 'train'),
                os.path.join(root, 'validation'),
                os.path.join(root, 'test_public'),
                ]
        elif phase == 'val':
            self.data_path = [
                os.path.join(root, 'test_public'),
                ]
        else:
            self.data_path = [
                os.path.join(root, 'test_private'),
                os.path.join(root, 'test_private_mixed'),
                ]
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        if phase == 'test':
            self.img_paths, self.gt_paths, self.labels, self.types = self.load_test_dataset()  # only return img_paths
        else:
            self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        for img_path in self.data_path:
            defect_types = os.listdir(img_path)
            if 'ground_truth' in defect_types:
                defect_types.remove('ground_truth')

            for defect_type in defect_types:
                if defect_type == 'good':
                    img_paths = glob.glob(os.path.join(img_path, defect_type) + "/*.png") + \
                                glob.glob(os.path.join(img_path, defect_type) + "/*.JPG") + \
                                glob.glob(os.path.join(img_path, defect_type) + "/*.bmp")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
                else:
                    if self.phase == 'train':  # 在训练时 test_public 中只载入 good 数据
                        continue
                    img_paths = glob.glob(os.path.join(img_path, defect_type) + "/*.png") + \
                                glob.glob(os.path.join(img_path, defect_type) + "/*.JPG") + \
                                glob.glob(os.path.join(img_path, defect_type) + "/*.bmp")
                    gt_paths = glob.glob(os.path.join(img_path, 'ground_truth', defect_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)
    
    def load_test_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        for img_path in self.data_path:
            img_paths = glob.glob(img_path + "/*.png") + \
                        glob.glob(img_path + "/*.JPG") + \
                        glob.glob(img_path + "/*.bmp")
            img_tot_paths.extend(img_paths)
        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # phase test

        # phase train and val
        img_path, gt_path, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt_path)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_path
