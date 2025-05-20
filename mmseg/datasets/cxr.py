import os
import cv2
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from IPython import embed
from .tools import ResizeAndPad, decode_mask


class CXR(Dataset):

    def __init__(self, root, img_size=1024):
        
        self.folder_path = root
        self.image_list = self._load_images()
        self.img_size = img_size
        self.transform = ResizeAndPad(img_size)

    def _load_images(self):  # get images
        image_list = []

        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.folder_path, filename)

                ground_truth_path = self.folder_path.replace("CXR_png", "masks")
                ground_truth_file_path = os.path.join(ground_truth_path, filename.replace(".png", "_mask.png"))
            
                if os.path.exists(ground_truth_file_path):

                    image_list.append(image_path)
                                
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__ori(self, index):

        image_path = self.image_list[index]
        file_path = image_path.split("/")[-1]

        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))

        file_name = os.path.basename(image_path)
        file_name = file_name.replace(".png", "_mask.png")

        ground_truth_path = self.folder_path.replace("CXR_png", "masks")

        ground_truth_file_path = os.path.join(ground_truth_path, file_name)
        ground_truth = cv2.imread(ground_truth_file_path)  # gt由0 / 255组成，255就是target
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        ground_truth = cv2.resize(ground_truth, (256, 256))
        contours, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


        # Plot each contour as individual images
        contour_images = []  # 轮廓，就是把每个部位单独拿出来。可以可视化一下看结果
        bounding_boxes = []
        for i, contour in enumerate(sorted_contours):
            # Create a blank image of the same size as the gt image
            if cv2.contourArea(contour) > 0: 
                contour_img = np.zeros_like(ground_truth)
                # Draw the contour on the blank image
                cv2.drawContours(contour_img, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
                contour_img = contour_img / 255.0
                # Append the contour image to the list
                contour_images.append(contour_img)

                # Get the bounding box of the contour for prompted SAM
                non_zero_pixels = np.where(contour_img)
                if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
                    x_min = np.min(non_zero_pixels[1])
                    y_min = np.min(non_zero_pixels[0])
                    x_max = np.max(non_zero_pixels[1])
                    y_max = np.max(non_zero_pixels[0])

                    bbox = np.array([x_min, y_min, x_max, y_max])
                    bounding_boxes.append(bbox)
        bounding_boxes = np.array(bounding_boxes)  # 轮廓相关的边界
        gray_img_normalized = ground_truth / 255.0
        threshold = 0.5
        ground_truth = np.where(gray_img_normalized > threshold, 1, 0)

        # return image, ground_truth, contour_images[0], contour_images[1]
        # return image, ground_truth
        return image, ground_truth, contour_images, bounding_boxes, file_path

    def __getitem__(self, index):

        image_path = self.image_list[index]
        file_path = image_path.split("/")[-1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        file_name = os.path.basename(image_path)
        file_name = file_name.replace(".png", "_mask.png")

        ground_truth_path = self.folder_path.replace("CXR_png", "masks")

        ground_truth_file_path = os.path.join(ground_truth_path, file_name)

        label = Image.open(ground_truth_file_path).convert('L')  # gt由0 / 255组成，255就是target
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
        

    