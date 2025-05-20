import torch
import numpy as np
import cv2
from PIL import Image
import os
from skimage.util import img_as_float
from skimage import color
import torch.nn.functional as F
import torchvision.transforms.functional as F_tr
from .drawing import Drawer, show_cam_on_image
import pydensecrf.densecrf as dcrf  # gcc版本不能用10以上垡编译成功


Fundus = ['Drishti_GS', 'ORIGA', 'REFUGE', 'REFUGE_Valid', 'RIM_ONE_r3']
Polyp = ['BKAI', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']


def process_img_clip(img_path, preprocess):
    # process image format to CLIP model
    img = Image.open(img_path)
    img = img.convert('RGB')
    crop_size = preprocess.transforms[1].size
    img = img.resize(crop_size)
    return img


# get clip-cam map
def get_cams(image, text, cam, input_size=None, model_domain='biomedclip'):
    if input_size is None:
        input_size = image.size
    return cam((image, text), 0, input_size, model_domain)


@torch.no_grad()
def clip_cam_score(image, text_prompt, model):
    # before tokenlizer(text)
    img_per_text, text_per_img = model(image.unsqueeze(0).cuda(), text_prompt.cuda())
    score = text_per_img.detach().cpu().item()
    return score


def get_cam_image(input_image, cam):
    cam_imgs = Drawer.overlay_cam_on_image(input_image, cam, use_rgb=True)
    cat_image = Drawer.concat([cam_imgs])
    return cat_image


def get_saliency_map(img_path, text_prompt, cam_wrapper, preprocess, tokenizer, save_result=False, img_size=(512, 512)):
    # 注意这里不能torch.no_grad()，因为需要gradient information
    # 理论上img是已经处理好的了。
    # 默认所有image都是一个prompt

    # # gt_bboxes = get_gt_boxes(text_prompt, file_name) if show_box else []
    # gt_bboxes = []  # no gt bboxes
    # # get cam
    # grayscale_cams = get_cams(img, text_prompt, cam)  # return is a array, cam weight
    #
    # # display interact with image
    # cat_image = get_cam_image(img, grayscale_cams)

    img = Image.open(img_path).convert("RGB")  # 因为preprocess的输入是Image不能是numpy或者tensor
    img = img.resize(img_size)  # 保证和外面的shape一样
    img_uint = np.asarray(img.copy())
    raw_size = img_size
    input_img = preprocess(img).unsqueeze(0).cuda()
    text_token = tokenizer(text_prompt).cuda()
    # get cam for prompt and overlay on input image
    cam = cam_wrapper.getCAM(input_img, text_token, raw_size, 0, model_domain='biomedclip')
    float_img = img_as_float(img)
    if len(float_img.shape) == 2:
        float_img = color.gray2rgb(float_img)
    cam_img = show_cam_on_image(float_img, cam, use_rgb=True)
    saliency_map = cam_img.copy()
    cam_img = Image.fromarray(cam_img)
    cat_img = Image.new('RGB', (raw_size[0] * 2, raw_size[1]))
    cat_img.paste(img, (0, 0))
    cat_img.paste(cam_img, (raw_size[0], 0))

    return cat_img, saliency_map, img_uint


# get crf map based cam
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_crf_map(img, annos, save=False, n_classes=2, gaussian_sxy=3, bilateral_sxy=60, bilateral_srgb=5, tau=1.05,
                epsilon=1e-8):
    # img和anno应该都是numpy输入。就是已经读取好的数据
    annos = np.squeeze(annos[:, :, 0])
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_classes)

    anno_norm = annos / 255.
    n_energy = -np.log((1.0 - anno_norm + epsilon)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + epsilon) / (tau * sigmoid(anno_norm))

    U = np.zeros((n_classes, img.shape[0] * img.shape[1]), dtype='float32')
    # print(U.shape, n_energy.shape, p_energy.shape)
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=gaussian_sxy, compat=3)
    d.addPairwiseBilateral(sxy=bilateral_sxy, srgb=bilateral_srgb, rgbim=img, compat=5)

    # Do the inference
    Q = d.inference(1)
    map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    # Save the output as image
    map *= 255
    if save:
        output = './a.jpg'
        cv2.imwrite(output, map.astype('uint8'))
    return map  # 注意是255


# memory bank parameters
def get_memory_bank_parameters(domain):
    if domain in Fundus:
        # [40, 16] ==> [RIM_ONE_r3:69.3, REFUGE:84.1, ORIGA: 55, REFUGE_Valid: 44, Drishti_GS:76.5, ]
        if domain == 'ORIGA':
            return [80, 32]
        elif domain == 'REFUGE_Valid':
            return [40, 8]
        else:
            return [40, 16]

    else:
        return [40, 16]


# SAM output
def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):  # 这里输入是0 / 255不是0和1
    _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, width, height]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return estimated_boxes, contours


@torch.no_grad()
def get_sam_output(predictor, img, mask):
    boxes, _ = scoremap2bbox(mask, 0, multi_contour_eval=False)
    predictor.set_image(img)  # 这里输入是numpy，不能是tensor
    boxes = np.array(boxes)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    # binary_mask
    masks = np.squeeze(masks).astype(float)  # squeeze其实就是取第一个
    return masks


# save result
def save_results(cam_img, crf_img, predict_mask, gt, path, file_name, crf_img_refine=None):
    # all are numpy format

    crf_img = np.stack((crf_img, crf_img, crf_img), axis=-1)
    predict_mask = np.stack((predict_mask, predict_mask, predict_mask), axis=-1)
    gt = np.stack((gt, gt, gt), axis=-1)
    # print(cam_img.shape, crf_img.shape, predict_mask.shape, gt.shape)
    img = np.hstack((cam_img, crf_img, predict_mask * 255, gt * 255))
    if crf_img_refine is not None:
        crf_img_refine = np.stack((crf_img_refine, crf_img_refine, crf_img_refine), axis=-1)
        img = np.hstack((img, crf_img_refine))
    cv2.imwrite(os.path.join(path, '{}.png'.format(file_name)), img)


def get_text_prompt(domain='fundus', baseline='SourceOnly'):
    print('Loading text prompts from {}!'.format(domain))
    if domain in Fundus:
        text_prompt = [
            # positive prompt
            'An image of a fundus photographs for glaucoma assessment',
            'A detailed fundus photograph highlighting features relevant for glaucoma analysis.',
            'A high-resolution image of the optic disc from a fundus camera for glaucoma screening.',
            'A fundus photography focusing on the optic nerve head for glaucoma evaluation.',
            'An optical fundus image capturing the retinal nerve fiber layer for assessing glaucoma risk.',
            'A clinical fundus photograph used for the detection of early-stage glaucoma changes in the optic nerve head.',
            'A fundus image with clear visibility of the optic cup and rim used for diagnosing progressive glaucoma.',
            'An image showing the optic nerve head with signs of glaucoma for clinical assessment purposes.',
            # negative prompt
            'A detailed view of the retinal blood vessels in a fundus photograph for vascular study.',
            'An image showcasing the complex network of blood vessels in the fundus, without focusing on the optic disc or macular region.',
            'A fundus photograph highlighting the vascular structure, suitable for analyzing blood vessel health and abnormalities.',
            'A fundus image primarily showing the macula and fovea, without details of the optic nerve head for glaucoma.',
            'A fundus photograph focused on assessing the overall retinal health, not specifically for glaucoma detection.',
            # 'An image of the retina displaying general eye conditions, not related to the glaucoma or optic nerve health.',
        ]
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        if baseline == 'SourceOnly':
            text_prompt = ['An image of a fundus photographs']

    elif domain == 'ISBI':
        text_prompt = [
            # positive prompt
            'An image of a melanoma detection in dermatological examinations',
            'A detailed dermoscopic photograph showing the distinct patterns associated with melanocytic lesions for melanoma screening.',
            'An image capturing the specific asymmetry, border irregularity, color variation, and diameter of a melanoma lesion.',
            'A clinical skin image documenting potential melanoma signs for early-stage skin cancer diagnosis.',
            'A dermoscopic image featuring key diagnostic features of melanoma used in expert dermatological assessments.',
            # negative prompt
            'An image of healthy skin without any visible lesions or markings, unrelated to skin cancer detection.',
            'A dermatological image showing common skin conditions such as acne or psoriasis, not related to melanoma or other skin cancers.',
            'A close-up image of the skin showing minor abrasions or scars, not indicative of melanoma or dermatological malignancies.',
        ]
        labels = [1, 1, 1, 1, 1, 0, 0, 0]

        if baseline == 'SourceOnly':
            text_prompt = ['An image of a melanoma detection in dermatological examinations']

    elif domain in Polyp:
        text_prompt = [
            # positive prompt
            'A detailed close-up image of a polyp visible during an endoscopy procedure.',
            'A clear and focused photograph of a polyp, useful for medical diagnosis in gastroenterology.',
            'High-resolution image of a colorectal polyp suitable for medical analysis and treatment planning.',
            'A clinical image of a polyp in the gastrointestinal tract aiding in early cancer detection.',
            'A well-defined polyp observed in a colonoscopy, crucial for patient treatment and care.',
            # negative prompt
            'An image of healthy gastrointestinal tissue without any signs of polyps.',
            'A photo of medical equipment used in an endoscopy room, unrelated to polyp detection.',
            'A general hospital setting with no focus on specific medical conditions or procedures.',
        ]
        labels = [1, 1, 1, 1, 1, 0, 0, 0]

        if baseline == 'SourceOnly':
            text_prompt = ['An image of a polyp photographs']

    elif domain == 'cxr':
        text_prompt = [
            'an image of a Lung Chest X-ray',
            "The X-ray evaluates the deviation of the trachea from its normal position, indicating potential mediastinal shift or mass effect on the lungs.",
            "The X-ray detects fine linear or reticular opacities within the lung fields, indicating interstitial lung disease.",
            "The X-ray examines the appearance and distribution of bronchi and blood vessels within the lungs.",
            "The X-ray examines the borders of the lungs, assessing for normal anatomy or potential abnormalities.",
            "The X-ray assesses the structures within the mediastinum, providing insights into their impact on the adjacent lungs."
        ]
        labels = [1, 1, 1, 1, 1, 1]

        if baseline == 'SourceOnly':
            text_prompt = ['an image of a Lung Chest X-ray']

    elif domain == 'BT_MRI':
        text_prompt = ['a malignant tumors image',
                       'An ultrasound scan image of breast cancer.',
                       ' A breast ultrasound image showing cancerous tissue.',
                       'An ultrasound medical image of a breast with cancer.',
                       'A diagnostic ultrasound image of breast cancer.',
                       'A sonographic image of breast cancer from an ultrasound.',
                       'a benign tumors image',
                       'a benign tumors image without malignant'
                       ]
        labels = [1, 1, 1, 1, 1, 1, 0, 0]

        if baseline == 'SourceOnly':
            text_prompt = ['an image of a brain tumor']

    elif domain == 'BUSI':
        text_prompt = ['a benign and malignant tumors image',
                       'An ultrasound scan image of breast cancer.',
                       'The malignant tumors image shows the mass oriented in an irregular pattern, not following the typical tissue planes.',
                       'The margins in the malignant tumors image are ill-defined and spiculated, suggesting invasive growth into surrounding tissues.',
                       'The malignant tumors image reveals an irregular and spiculated shape, which is a common characteristic of malignancy.',
                       'The malignant tumors image displays hypoechoic regions, indicating lower echogenicity compared to normal tissue.',
                       "The malignant tumors image does not show clear and smooth boundaries typical of benign conditions.",
                       "The malignant tumors image lacks uniform texture, unlike benign lesions that usually have a homogeneous appearance."
                       ]

        labels = [1, 1, 1, 1, 1, 1, 0, 0]

        if baseline == 'SourceOnly':
            text_prompt = ['a benign and malignant tumors image']


    return text_prompt, labels


# augmentation
def get_aug_data(tensor, mode=''):
    tensor = F_tr.adjust_brightness(tensor, brightness_factor=torch.tensor(1.0) + torch.randn(1) * 0.2)
    tensor = F_tr.adjust_contrast(tensor, contrast_factor=1.5)
    tensor = F_tr.adjust_saturation(tensor, saturation_factor=1.3)
    tensor = F_tr.adjust_hue(tensor, hue_factor=0.1)


    return tensor


def rotate_image_tensor_2D(image_np, angle=90):
    # 将 Tensor 转换为 NumPy 数组
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.numpy()

    # 获取图像的中心
    center = (image_np.shape[1] // 2, image_np.shape[0] // 2)
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 执行旋转，使用最近邻插值以保留二值特性
    rotated_np = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]), flags=cv2.INTER_NEAREST)

    # 将旋转后的图像转换回 Tensor
    rotated_tensor = torch.from_numpy(rotated_np)
    return rotated_tensor


def rotate_image_tensor_3D(tensor, angle):
    # 确保输入 tensor 是 3D (C, H, W)
    assert tensor.dim() == 3 and tensor.size(0) == 3, "Input tensor must be 3D with shape (3, H, W)"

    # 将 Tensor 转换为 NumPy 数组
    image_np = tensor.numpy()

    # 初始化旋转后的 NumPy 数组
    rotated_np = np.zeros_like(image_np)

    # 获取图像的中心
    H, W = image_np.shape[1], image_np.shape[2]
    center = (W // 2, H // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 对每个通道分别进行旋转
    for i in range(3):
        rotated_np[i] = cv2.warpAffine(image_np[i], M, (W, H), flags=cv2.INTER_NEAREST)

    # 将旋转后的图像转换回 Tensor
    rotated_tensor = torch.from_numpy(rotated_np)

    return rotated_tensor


# 训练的时候用哪个超参数，这里主要是针对Fundus来说，因为其他数据集我还没来得及调试。

def get_best_parameters(domain):
    # params = [alpha, lr, lora_block, memory_size, neighbor, threshold]
    params = [0.9, 1e-6, 5, 40, 16, 128]  # default params
    if domain == 'RIM_ONE_r3':
        params = [0.9, 1e-5, 3, 40, 16, 128]
    elif domain == 'Drishti_GS':
        params = [0.999, 1e-6, 5, 40, 16, 128]
    elif domain == 'ORIGA':
        params = [0.9, 8e-6, 5, 40, 32, 100]
    elif domain == 'REFUGE':
        params = [0.999, 1e-6, 3, 40, 16, 100]
    elif domain == 'REFUGE_Valid':
        params = [0.999, 1e-5, 5, 80, 16, 128]
    elif domain == 'cxr':
        params = [0.999, 8e-6, 1, 80, 32, 50]

    return params



