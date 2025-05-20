import os

import numpy as np
import cv2
import torch
from PIL import Image
import clip
import torch.nn.functional as F
from torchvision.utils import save_image


def get_crops(image, masks, prompt_mode='crops'):
    # 确保了边界框的宽度和高度都是正数，即右下角坐标必须大于左上角坐标。如果不满足这个条件，边界框就是无效的
    imgs_bboxes = []
    indices_to_remove = []

    for i, mask in enumerate(masks):
        box = mask["bbox"]
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        if x2 > x1 and y2 > y1:  # Check if the bounding box has non-zero dimensions

            if prompt_mode == "crops":
                # crops
                # 利用segmentation得到mask
                seg_mask = np.array(
                    [mask["segmentation"], mask["segmentation"], mask["segmentation"]]).transpose(1, 2, 0)
                cropped_image = np.multiply(image, seg_mask).astype("int")[int(y1):int(y2), int(x1):int(x2)]
                imgs_bboxes.append(cropped_image)

            elif prompt_mode == "crop_expand":
                # crops
                seg_mask = np.array([mask["segmentation"], mask["segmentation"], mask["segmentation"]]).transpose(1 ,2
                                                                                                                  ,0)
                # Expand bounding box coordinates
                x1_expanded = max(0, x1 - 10)
                y1_expanded = max(0, y1 - 10)
                x2_expanded = min(image.shape[1], x2 + 10)
                y2_expanded = min(image.shape[0], y2 + 10)

                if x2_expanded > x1_expanded and y2_expanded > y1_expanded:
                    cropped_image = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    imgs_bboxes.append(cropped_image)


        else:
            print("Skipping zero-sized bounding box.", box, len(masks), i)
            indices_to_remove.append(i)

    for index in sorted(indices_to_remove, reverse=True):
        masks.pop(index)  # 这个mask del后好像在其他函数之后就不影响了，所以后面可能得再del一次
    return imgs_bboxes, indices_to_remove, masks


def retrieve_relevant_crop(crops, class_names, model, preprocess, dataset):  # 与prompt算similarity
    # 第几个crop和text prompt 最相关
    crops_uint8 = [image.astype(np.uint8) for image in crops]

    pil_images = []
    for image in crops_uint8:
        if image.shape[0] > 0 and image.shape[1] > 0:
            pil_image = Image.fromarray(image)
            pil_images.append(pil_image)  # 保存每个小块
    # 把每个小块都做处理，这里注意是resize到原来大小，或者normalization
    preprocessed_images = [preprocess(image).to("cuda") for image in pil_images]
    stacked_images = torch.stack(preprocessed_images)  # 把sam得到的crop丢到clip中
    # stacked_images.shape = [n_crops, 3, h, w]
    similarity_scores = {class_name: [] for class_name in class_names}  # class_names就是text prompt

    with torch.no_grad():
        # 计算这么多个corp和text prompt的similarity sorce
        # [n, dim]
        image_features = model.encode_image(stacked_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        for class_name in class_names:
            class_descriptions = class_names[class_name]
            class_text_features = [model.encode_text(clip.tokenize(description).to("cuda")) for description in
                                   class_descriptions]

            mean_text_feature = torch.mean(torch.stack(class_text_features), dim=0)
            mean_text_feature /= mean_text_feature.norm(dim=-1, keepdim=True)

            similarity_score = 100. * image_features @ mean_text_feature.T
            similarity_scores[class_name] = similarity_score.squeeze().tolist()
            # len similarity_scores 一共有n个source，我们这里只有一个class。这里就是把每个class的得到的区域算similarity。选出最大的mask crop

        # print(similarity_scores)
        # 只取source 最大的那个prop， key只有1个class，所以就是看这个class最大的crop
        if dataset == "cxr":
            max_indices = {
                key: sorted(range(len(similarity_scores[key])), key=lambda i: similarity_scores[key][i], reverse=True)[
                     :2] for key in similarity_scores}  # 2 lungs so getting top 2

        else:
            # 这里如果只有1个符合的crop，max出来是一个浮点数，好奇怪
            try:
                max_indices = {key: similarity_scores[key].index(max(similarity_scores[key])) for key in
                               similarity_scores}
            except:
                max_indices = {class_name: 0}  # 只有1个
            # max_indices = {key: sorted(range(len(similarity_scores[key])), key=lambda i: similarity_scores[key][i], reverse=True)[:2] for key in similarity_scores} # 2 lungs so getting top 2
        # print(max_indices, similarity_scores)

    if not isinstance(max_indices[class_name], list):
        max_indices[class_name] = [max_indices[class_name]]
    return max_indices, similarity_scores


def get_sam_prompts(image, masks, max_indices, imgs_bboxes):
    # 在图像上生成并绘制目标物体的边界框，并将这些边界框信息和相关的裁剪区域返回
    # ------  bbox prompts cordinates relevant to ROI for SAM------

    bboxes = []
    relevant_crop = []
    img_with_bboxes = []
    for key, indices in max_indices.items():
        for index, value in enumerate(indices):
            bbox = masks[value]["bbox"]
            bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            if index == 0:
                img = image.copy()

            img_with_bboxes = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
            bboxes.append(bbox)
            relevant_crop.append(imgs_bboxes[value])

    bboxes = np.array(bboxes)
    return bboxes, relevant_crop, img_with_bboxes


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image)
    return image.permute(2, 0, 1).contiguous()


def sam_predicton(sam, image, resize_transform, bboxes, dataset, mode='sam_clip'):

    # ------ SAM format ------

    batched_input = [{
        'image': prepare_image(image.astype(np.uint8), resize_transform, "cuda").to("cuda"),  # [3, 1024, 1024]
        'boxes': resize_transform.apply_boxes_torch(torch.from_numpy(np.array(bboxes)), image.shape[:2]).to("cuda"),
        'original_size': image.shape[:2]
    }]
    # print('print image')
    # print(batched_input[0]['image'])
    # [2, 1, 256,256]  算sigmoid然后判断
    # preds = sam(batched_input, multimask_output=False)  # forward result
    preds = sam(batched_input, multimask_output=False)[0]['masks']  # forward result,这里返回是一个list所以要注意

    binary_masks = torch.sigmoid(preds) > 0.5
    binary_masks = binary_masks.squeeze().cpu().numpy()

    if dataset == "cxr" and mode == "sam_clip" or mode == "sam_prompted":
        binary_masks = np.bitwise_or(binary_masks[0], binary_masks[1])  # 为运算。这里为啥是两个mask，cxr左右两个肺

    return binary_masks


def dice_coeff(truth, prediction):
    # if len(prediction > 1):
    #     # for lungs
    #     prediction = np.bitwise_or(prediction[0], prediction[1])
    # dice
    if len(truth.shape) != 2:
        pass
    # print(np.unique(truth), np.unique(prediction))

    intersection = np.sum(truth * prediction)
    union = np.sum(truth) + np.sum(prediction)
    dice = 2.0 * intersection / union

    # mIoU
    iou = np.mean(intersection / (union - intersection))

    # precision
    total_pixel_pred = np.sum(prediction)
    precision = np.mean(intersection / total_pixel_pred)

    # recall
    total_pixel_truth = np.sum(truth)
    recall = np.mean(intersection / total_pixel_truth)

    return dice, iou


def dice_loss_fn(truths, predictions):
    pass


def iFFT(amp_src_, pha_src, imgH, imgW):
    # recompose fft
    real = torch.cos(pha_src) * amp_src_
    imag = torch.sin(pha_src) * amp_src_
    fft_src_ = torch.complex(real=real, imag=imag)

    src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
    return src_in_trg


def FFT_image(image, prompt_alpha=0.5, prompt_type='lowpass', rate=0.5):
    prompt_size = int(image.shape[0] * prompt_alpha)
    padding_size = (image.shape[0] - prompt_size)//2
    data_prompt = torch.ones((1, 3, prompt_size, prompt_size))
    # image.shape = [H, W, 3]
    imgH, imgW, _ = image.shape
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # 不需要除以255
    fft = torch.fft.fft2(x.clone().float(), dim=(-2, -1))
    # extract amplitude and phase of both ffts
    amp_src, pha_src = torch.abs(fft), torch.angle(fft)
    amp_src = torch.fft.fftshift(amp_src)

    # obtain the low frequency amplitude part
    prompt = F.pad(data_prompt,
                   [padding_size, imgH - padding_size - prompt_size, padding_size,
                    imgW - padding_size - prompt_size],
                   mode='constant', value=1.0).contiguous()

    amp_src_ = amp_src * prompt
    amp_src_ = torch.fft.ifftshift(amp_src_)

    amp_low_ = amp_src[:, :, padding_size: padding_size + prompt_size,
               padding_size: padding_size + prompt_size]

    inv = iFFT(amp_src_, pha_src, imgH, imgW)

    # mask = torch.zeros(x.shape).to(x.device)
    # w, h = x.shape[-2:]
    # line = int((w * h * rate) ** .5 // 2)
    # mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1
    #
    # fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    #
    # if prompt_type == 'highpass':  # high component
    #     fft = fft * (1 - mask)
    # elif prompt_type == 'lowpass':  # low component
    #     fft = fft * mask
    # fr = fft.real
    # fi = fft.imag
    #
    # fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    # inv = torch.fft.ifft2(fft_hires, norm="forward").real
    #
    # inv = torch.abs(inv)

    # save_image(x.float(), './a.jpg', )
    # save_image(inv.float(), './c.jpg',)  # 这里的FFT没有其他CS那么明显
    # print(inv.squeeze(0).permute(1, 2, 0).contiguous().shape)
    return inv.squeeze(0).permute(1, 2, 0).contiguous().numpy()


def index_selection_by_gt(masks, image, gt, img_crops, resize_transform, sam, indices_to_remove, return_info=False):

    dice_scores, mious = [], []
    result_masks = []
    # 之前筛选出来的crop，有一些mask的crop不符合，所以这部分的信息不运算，要把这个index去掉
    # for index in sorted(indices_to_remove, reverse=True):
    #     # 前面get_crop del的对这里没影响，所以这里得继续del一次！！
    #     del masks[index]
    indices = {'class': [i for i in range(len(masks))]}
    bboxes, relevant_crop, img_with_bboxes = get_sam_prompts(image, masks, indices, img_crops)
    for bbox in bboxes:
        batched_input = [{
            'image': prepare_image(image.astype(np.uint8), resize_transform, "cuda").to("cuda"),  # [3, 1024, 1024]
            'boxes': resize_transform.apply_boxes_torch(torch.from_numpy(np.array(bbox)), image.shape[:2]).to("cuda"),
            'original_size': image.shape[:2]
        }]
        # [2, 1, 256,256]  算sigmoid然后判断
        # 这里注意一个image返回一个mask，所以是一个list
        preds = sam(batched_input, multimask_output=False)[0]['masks']  # forward result
        binary_masks = torch.sigmoid(preds) > 0.5
        binary_masks = binary_masks.squeeze().cpu().numpy()
        dice_score, miou = dice_coeff(gt, binary_masks)
        dice_scores.append(dice_score)
        mious.append(miou)
        result_masks.append(binary_masks)

    if len(dice_scores) == 0:
        return 0, result_masks[0]
    else:
        if return_info:
            return np.argmax(dice_scores), result_masks[np.argmax(dice_scores)], dice_scores, mious
        else:
            return np.argmax(dice_scores), result_masks[np.argmax(dice_scores)]


def save_image_to_path(file_path, f_name, img, numpy_format=True):
    print(file_path)
    os.makedirs(file_path, exist_ok=True)
    if numpy_format:
        print(os.path.join(file_path, '{}.jpg'.format(f_name)),)
        cv2.imwrite(os.path.join(file_path, '{}.jpg'.format(f_name)), img)
