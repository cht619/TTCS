
import numpy as np
import torch
import cv2
from torch.utils import data
import pandas as pd
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib.pyplot as plt
from .tools import ResizeAndPad, decode_mask



Polyp = ['BKAI', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']



class Polyp_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, img_size=1024, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.image_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (img_size, img_size)
        self.img_normalize = img_normalize
        self.transform = ResizeAndPad(img_size)

    def __len__(self):
        return self.len
    def __getitem__(self, index):

        if self.label_list[index].endswith('tif'):
            self.label_list[index] = self.label_list[index].replace('.tif', '-{}.tif'.format(1))
        image_path = os.path.join(self.root, self.image_list[index])
        ground_truth_path = os.path.join(self.root, self.label_list[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to RGB
        label = Image.open(ground_truth_path).convert('L')
        gt_mask = np.array(label) * 255  # 这里不需要转换，直接255白色就是目标区域

        masks = []
        bboxes = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            masks.append(mask)
            x, y, w, h = cv2.boundingRect(mask)  # get bbox prompt
            bboxes.append([x, y, x + w, y + h])

        # 这里注意mask输入是一个list
        image, gt_masks, bboxes = self.transform(image=image, masks=masks, bboxes=np.asarray(bboxes))  # 这里主要就是resize
        gt_masks = np.stack(gt_masks, axis=0)
        bboxes = np.stack(bboxes, axis=0)

        return {'img': image, 'gt': torch.tensor(gt_masks)[0], 'bboxes': torch.tensor(bboxes)[0], 'file': image_path}

