import numpy as np
import torch
import cv2
import h5py
import os
from torch.utils import data
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from .tools import ResizeAndPad, decode_mask


# 针对UDIAT的数据，有四个dir：Benign/       Benign_mask/    Malignant/     Malignant_mask/
class BUSI_dataset(data.Dataset):
    def __init__(self, root, img_size=1024):
        self.root = root
        self.img_size = img_size
        self.domains = ['Benign', 'Malignant']
        self.files = []
        for domain in self.domains:
            root_domain = os.path.join(self.root, domain)
            for file in os.listdir(root_domain):
                if file.endswith('.png'):
                    file_path = os.path.join(root_domain, file)
                    self.files.append(file_path)
        self.transform = ResizeAndPad(img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        if self.domains[0] in image_path:
            ground_truth_path = image_path.replace(self.domains[0], '{}_mask'.format(self.domains[0]))
        else:
            ground_truth_path = image_path.replace(self.domains[1], '{}_mask'.format(self.domains[1]))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to RGB
        label = Image.open(ground_truth_path).convert('L')
        # label = label.resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)
        gt_mask = np.array(label) * 255

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




