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


# https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/code


class BTMRI_dataset(data.Dataset):
    def __init__(self, root, img_size=1024):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.files = []
        for file in os.listdir(root):
            if file.endswith('.png') and 'mask' not in file:
                file_path = os.path.join(root, file)
                self.files.append(file_path)
        self.transform = ResizeAndPad(img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        ground_truth_path = image_path.replace('.png', '_mask.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to RGB
        label = Image.open(ground_truth_path).convert('L')
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




def read_mat_data_to_img(root):
    save_path = root.replace('/dataset/data', '/dataset/images_masks')
    for file in tqdm(os.listdir(root)):
        file_path = os.path.join(root, file)
        with h5py.File(file_path, 'r') as f:
            # mask 原本只有0和1，这里乘以255
            image, mask = f['cjdata']['image'][:], f['cjdata']['tumorMask'][:] * 255
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, axis=2) / 3075 * 255
            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(file.split('.')[0])), image.astype(np.uint8))
            # cv2.imwrite(os.path.join(save_path, '{}_mask.png'.format(file.split('.')[0])), mask.astype(np.uint8))
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_path, '{}.png'.format(file.split('.')[0])))
            Image.fromarray(mask.astype(np.uint8)).save(os.path.join(save_path, '{}_mask.png'.format(file.split('.')[0])))



if __name__ == '__main__':
    root = '/mnt/sda/Dataset/Image_DA/Segmentation/MedicalSegmentation/brain_tumor_mri/dataset/data'
    read_mat_data_to_img(root)


    # /mnt/sda/Dataset/Image_DA/Segmentation/MedicalSegmentation/brain_tumor_mri/dataset
    # /mnt/sda/Dataset/Image_DA/Segmentation/MedicalSegmentation/brain_tumor_mri/images_masksset/images_masks