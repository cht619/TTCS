import copy
import torch
import cv2
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib.pyplot as plt
from .tools import ResizeAndPad, decode_mask


class ISBI_dataset(data.Dataset):   # 2016
    def __init__(self, root_dir, df, img_size=1024, if_self_training=False):
        super().__init__()
        # df = pd.read_csv(df, encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.root_dir = root_dir
        self.transform = ResizeAndPad(img_size)  # 1024就是sam的输入

        self.if_self_training = if_self_training  # 如果是training data就是True

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.root_dir, name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = self.label_list[index]
        gt_path = os.path.join(self.root_dir, label_name)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
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
        # image = self.transform(image=image)  # 这里主要就是resize

        gt_masks = np.stack(gt_masks, axis=0)
        bboxes = np.stack(bboxes, axis=0)

        return {'img': image, 'gt': torch.tensor(gt_masks), 'bboxes': torch.tensor(bboxes), 'file': image_path}
