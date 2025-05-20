
import torch
import cv2
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib.pyplot as plt
from .tools import ResizeAndPad, decode_mask
# from ..segment_anything import ResizeLongestSide  # 这里有问题，到时再看怎么优化


# OPTIC dataset


Fundus = ['Drishti_GS', 'ORIGA', 'REFUGE', 'REFUGE_Valid', 'RIM_ONE_r3']


class OPTIC_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, img_size=1024, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.image_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (img_size, img_size)
        self.img_normalize = img_normalize
        self.transform = ResizeAndPad(img_size)
        # if batch_size is not None:
        #     iter_nums = len(self.img_list) // batch_size
        #     scale = math.ceil(100 / iter_nums)
        #     self.img_list = self.img_list * scale
        #     self.label_list = self.label_list * scale

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
        label_npy = np.array(label)
        gt_mask = np.zeros_like(label_npy)
        # 0是是黑，255是白色
        gt_mask[label_npy == 255] = 0  # 原来的白色是背景，所以把他们都变成黑色
        gt_mask[label_npy == 128] = 1
        gt_mask[label_npy == 0] = 1

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

        # contours, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        #
        # # Plot each contour as individual images
        # contour_images = []  # 轮廓，就是把每个部位单独拿出来。可以可视化一下看结果
        # bounding_boxes = []
        # for i, contour in enumerate(sorted_contours):
        #     # Create a blank image of the same size as the gt image
        #     if cv2.contourArea(contour) > 0:
        #         contour_img = np.zeros_like(ground_truth)
        #         # Draw the contour on the blank image
        #         cv2.drawContours(contour_img, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
        #         contour_img = contour_img / 255.0
        #         # Append the contour image to the list
        #         contour_images.append(contour_img)
        #
        #         # Get the bounding box of the contour for prompted SAM
        #         non_zero_pixels = np.where(contour_img)
        #         if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
        #             x_min = np.min(non_zero_pixels[1])
        #             y_min = np.min(non_zero_pixels[0])
        #             x_max = np.max(non_zero_pixels[1])
        #             y_max = np.max(non_zero_pixels[0])
        #
        #             bbox = np.array([x_min, y_min, x_max, y_max])
        #             bounding_boxes.append(bbox)
        # bounding_boxes = np.array(bounding_boxes)  # 轮廓相关的边界
        # gray_img_normalized = ground_truth / 255.0
        # threshold = 0.5
        # ground_truth = np.where(gray_img_normalized > threshold, 1, 0)
        #
        # # return image, ground_truth, contour_images[0], contour_images[1]
        # # return image, ground_truth
        # # 不过一般就是用image和gt
        # return image, ground_truth, contour_images, bounding_boxes, image_path


def get_bounding_box(ground_truth_map: np.array) -> list:
    """
    Get the bounding box of the image with the ground truth mask

      Arguments:
          ground_truth_map: Take ground truth mask in array format

      Return:
          bbox: Bounding box of the mask [X, Y, X, Y]

    """
    # get bounding box from mask
    idx = np.where(ground_truth_map > 0)
    x_indices = idx[1]
    y_indices = idx[0]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class Samprocessor:
    """
    Processor that transform the image and bounding box prompt with ResizeLongestSide and then pre process both data
        Arguments:
            sam_model: Model of SAM with LoRA weights initialised

        Return:
            inputs (list(dict)): list of dict in the input format of SAM containing (prompt key is a personal addition)
                image: Image preprocessed
                boxes: bounding box preprocessed
                prompt: bounding box of the original image

    """

    def __init__(self, size=1024):
        super().__init__()
        # self.model = sam_model
        self.transform = ResizeLongestSide(size)
        self.reset_image()

    def __call__(self, image: Image, original_size: tuple, prompt: list) -> dict:
        # Processing of the image
        image_torch = self.process_image(image, original_size)

        # Transform input prompts
        box_torch = self.process_prompt(prompt, original_size)  # 这里事实上就是4个点，然后reshape成sam的大小

        inputs = {"image": image_torch,
                  "original_size": original_size,
                  "boxes": box_torch,
                  "prompt": prompt}

        return inputs

    def process_image(self, image: Image, original_size: tuple) -> torch.tensor:
        """
        Preprocess the image to make it to the input format of SAM

        Arguments:
            image: Image loaded in PIL
            original_size: tuple of the original image size (H,W)

        Return:
            (tensor): Tensor of the image preprocessed
        """
        nd_image = np.array(image)
        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return input_image_torch

    def process_prompt(self, box: list, original_size: tuple) -> torch.tensor:
        """
        Preprocess the prompt (bounding box) to make it to the input format of SAM

        Arguments:
            box: Bounding bounding box coordinates in [XYXY]
            original_size: tuple of the original image size (H,W)

        Return:
            (tensor): Tensor of the prompt preprocessed
        """
        # We only use boxes
        box_torch = None
        nd_box = np.array(box).reshape((1, 4))
        box = self.transform.apply_boxes(nd_box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        box_torch = box_torch[None, :]

        return box_torch

    @property
    def device(self) -> torch.device:
        return torch.device('cuda')

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None



class OPTIC_dataset_FT(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=512, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.image_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.img_size = (target_size, target_size)
        self.img_normalize = img_normalize
        # if batch_size is not None:
        #     iter_nums = len(self.img_list) // batch_size
        #     scale = math.ceil(100 / iter_nums)
        #     self.img_list = self.img_list * scale
        #     self.label_list = self.label_list * scale
        self.processor = Samprocessor()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.label_list[index].endswith('tif'):
            self.label_list[index] = self.label_list[index].replace('.tif', '-{}.tif'.format(1))
        image_path = os.path.join(self.root, self.image_list[index])
        ground_truth_path = os.path.join(self.root, self.label_list[index])

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]))  # 这里输出最终结果
        label = Image.open(ground_truth_path).convert('L')
        label = label.resize(self.img_size, resample=Image.NEAREST)
        label_npy = np.array(label)
        mask = np.zeros_like(label_npy)
        # 0是是黑，255是白色
        mask[label_npy == 255] = 0  # 原来的白色是背景，所以把他们都变成黑色
        mask[label_npy == 128] = 1
        mask[label_npy == 0] = 1

        box = get_bounding_box(mask)  # 这里处理也是大于0的像素，即不是黑色的像素
        inputs = self.processor(image, (self.img_size[0], self.img_size[1]), box)
        inputs["ground_truth_mask"] = torch.from_numpy(mask)
        inputs['filename'] = image_path
        return inputs


def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset

    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)