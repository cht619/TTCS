import cv2
import copy
import os.path
import csv
import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import save_image
import torch.nn.functional as F
import wandb
from numpy.linalg import norm
from PIL import Image
from tqdm import tqdm
from ..saliency_maps.clip_model import load_clip_cam
from .sourceonly import SourceOnly, CLIP_TTT, CHECK_NUM_PARAMS, build_sam_predictor
from .utils import (process_img_clip, get_saliency_map, get_crf_map, get_sam_output, get_text_prompt, save_results,
                    scoremap2bbox, get_memory_bank_parameters, rotate_image_tensor_2D, rotate_image_tensor_3D,
                    get_aug_data, get_best_parameters)
from ..models.coop_clip import ClipTestTimeTuning, ClipTestTimeTuning_VP
from .sam_utils import dice_coeff
from ..segment_anything.sam_lora import LoRA_Sam, SAM_Model


# 简单的的flip augmentation
def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]


class Memory(object):
    """
        Create the empty memory buffer. save crf map
    """

    def __init__(self, size, dimension=1024*1024):
        self.memory = {}
        self.size = size
        self.dimension = dimension  # SAM的输入大小

    def get_size(self):
        return len(self.memory)

    def push(self, keys, crf_map):  # crf_map.shape=[1024*1024]
        for i, key in enumerate(keys):
            if len(self.memory.keys()) > self.size:
                self.memory.pop(list(self.memory)[0])  # FIFO

            self.memory.update(
                {key.reshape(self.dimension).tobytes(): crf_map})

    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples).squeeze().numpy()


class TTA_base(SourceOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_model, self.preprocess, self.tokenizer, self.cam = load_clip_cam(use_visual_prompt=True)
        self.clip_model = self.cam.model  # 模型就是有prompt的

        memory_parameters = get_memory_bank_parameters(self.cfg.domain)
        # self.memory_bank = Memory(size=memory_parameters[0], dimension=self.clip_model.visual_prompt.data_prompt.numel())
        # self.neighbor = memory_parameters[1]  # 8 or 16 再进一步验证。 ORIGA设置为32，size是80，threshold100，alpha0.9效果不错！！ 其他应该不用这么高

        self.memory_bank = Memory(size=32, dimension=self.clip_model.visual_prompt.data_prompt.numel())
        self.neighbor = 16

    def build_optimizer(self, mode='prompt', lr=7e-3):
        # 这里是不是应该是两个部分？CLIp
        if mode == 'prompt':
            parameters = self.clip_model.visual_prompt.parameters()
            for p in self.clip_model.model.parameters():  # freeze CLIP
                p.requires_grad = False

        optimizer_clip = optim.Adam(parameters, lr=lr, weight_decay=5e-4)

        # SAM
        CHECK_NUM_PARAMS(self.clip_model, lr)
        return optimizer_clip

    def refine_crf_map_memory_map(self, crf_img, img, threshold=50, alpha=0.9):  # 128是相对稳定的参数
        # 其他都是0.999，就ORIGA效果不行，得不断调整
        _, low_component = self.clip_model.visual_prompt(img.cuda())
        # get neighbour.
        if len(self.memory_bank.memory.keys()) >= self.neighbor:
            crf_img_memory = self.memory_bank.get_neighbours(keys=low_component.cpu().numpy(), k=self.neighbor)
            crf_img = alpha * crf_img_memory + (1 - alpha) * crf_img
            # crf_img = crf_img_memory

        self.memory_bank.push(keys=low_component.cpu().numpy(), crf_map=crf_img)
        crf_img = (crf_img > threshold) * 255
        return crf_img.astype(np.uint8)

    def forward_clip(self, file_paths, optimizer, file_name=None):
        # file_paths输入是个list，至少
        data = torch.stack([self.preprocess(Image.open(file)) for file in file_paths]).cuda()
        image_logits, text_logits = self.clip_model(data, self.tokenizer(self.text_prompt).cuda())
        loss = F.binary_cross_entropy_with_logits(image_logits, self.labels.unsqueeze(0)) / len(self.text_prompt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if image_logits.max().data < 34:
            self.clip_model.visual_prompt.reset_prompt()  # 这里的reset为1好像也太不adaptive了

    def forward_step(self, data, optimizer, save=True, memory_bank=True):
        # 保证img gt bbox的shape都是[bs, xxx]  第一维是bs
        img, gt, bbox, file_path = data['img'], data['gt'], data['bboxes'], data['file'][0]  # bs=1
        file_name = file_path.split('/')[-1][:-4]
        self.forward_clip([file_path], optimizer)
        torch.cuda.empty_cache()
        # get result. 这里注意如果只用positive text prompt效果是不是会好一些
        cat_img, saliency_map, img_np_uint = get_saliency_map(
            file_path, self.text_prompt[0], self.cam, self.preprocess, self.tokenizer, img_size=img.shape[-2:])
        crf_img = get_crf_map(img_np_uint.copy(), saliency_map)  # numpy format

        # memory_bank
        if memory_bank:
            crf_img_refine = self.refine_crf_map_memory_map(crf_img, img)
            masks = get_sam_output(self.sam_predictor, img_np_uint, crf_img_refine)
        else:
            masks = get_sam_output(self.sam_predictor, img_np_uint, crf_img)
            crf_img_refine = None
        if save:
            save_path = os.path.join(self.cfg.work_dir, 'Output')
            os.makedirs(save_path, exist_ok=True)
            save_results(saliency_map, crf_img, masks, gt.squeeze().numpy(), save_path, file_name, crf_img_refine)
        return masks

    def forward(self):
        optimizer = self.build_optimizer()
        dice_scores = []
        with tqdm(total=len(self.dataloader), position=0) as pbar:
            for n, data in enumerate(self.dataloader):
                if data['gt'].shape[-2:] != data['img'].shape[-2:]:  # 为什么会有一个图片不一样
                    data['gt'] = F.interpolate(
                        data['gt'].unsqueeze(1).float(), data['img'].shape[2:], mode='bilinear').squeeze(1).long()
                preds = self.forward_step(data, optimizer)
                pbar.update(1)
                dice_score, miou = dice_coeff(data['gt'].squeeze().numpy(), preds)
                dice_scores.append((dice_score, miou))
                pbar.set_description('[Process:{}.Dice:{:.2f}.mIOU:{:.2f}]'.format(
                    data['file'][0].split('/')[-1][:-4], dice_score, miou))

                with open(os.path.join(self.cfg.work_dir, 'result.csv'), 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([data['file'][0].split('/')[-1][:-4], dice_score, miou])
        dice_scores = np.array(dice_scores)
        average_dice_score, miou = np.mean(dice_scores[:, 0]), np.mean(dice_scores[:, 1])
        print("Average Dice Score:", average_dice_score, "mIoU:", miou)
        with open(os.path.join(self.cfg.work_dir, 'result.csv'), 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(['Avg Dice and mIoU', average_dice_score, miou])


# 结合更新SAM
class TTCS(TTA_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sam_lora = LoRA_Sam(copy.deepcopy(self.sam_predictor.model), r=4).cuda()
        # self.sam_lora = SAM_Model(copy.deepcopy(self.sam_predictor.model)).cuda()

        memory_parameters = get_memory_bank_parameters(self.cfg.domain)
        self.memory_bank = Memory(size=memory_parameters[0], dimension=self.clip_model.visual_prompt.data_prompt.numel())
        self.neighbor = memory_parameters[1]

    def build_optimizer(self, mode='prompt', lr=7e-3):
        if mode == 'prompt':
            parameters = self.clip_model.visual_prompt.parameters()
            for p in self.clip_model.model.parameters():  # freeze CLIP
                p.requires_grad = False

        optimizer_clip = optim.Adam(parameters, lr=lr, weight_decay=5e-4)
        # optimizer_sam = optim.Adam(self.sam_lora.parameters(), lr=1e-6, weight_decay=1e-4)
        optimizer_sam = optim.Adam(self.sam_lora.parameters(), lr=8e-6, weight_decay=1e-4)

        # SAM
        print('---------------CLIP----------------')
        CHECK_NUM_PARAMS(self.clip_model, lr)
        print('---------------SAM----------------')
        CHECK_NUM_PARAMS(self.sam_lora, lr)
        return optimizer_clip, optimizer_sam

    def forward_sam(self, img, prompt, optimizer, file_name=None):
        # 这里主要最后的的mask要用threshold设置下
        boxes, _ = scoremap2bbox(prompt, 0, multi_contour_eval=False)
        img_embeddings, pred_masks, ious, res_masks = self.sam_lora(img.cuda(), [torch.tensor(boxes).cuda()])

        # img_aug = rotate_image_tensor_3D(img.squeeze(0), angle=90)
        img_aug = get_aug_data(img)
        boxes_aug, _ = scoremap2bbox(prompt, 0, multi_contour_eval=False)
        img_embeddings_aug, pred_masks_aug, ious_aug, res_masks_aug = self.sam_lora(
            img_aug.cuda(), [torch.tensor(boxes_aug).cuda()])

        pred_logits, pred_logits_flip = pred_masks[0], pred_masks_aug[0]
        logits_avg = (pred_logits + pred_logits_flip) / 2

        probs_avg = F.softmax(logits_avg, dim=-1)
        probs = F.softmax(pred_logits, dim=-1)
        loss = F.kl_div(probs.log(), probs_avg, reduction='batchmean')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        masks = torch.cat((pred_logits, pred_logits_flip), 0).mean(0).detach().cpu().numpy()
        masks = (masks > self.sam_lora.lora_vit.mask_threshold).astype(float)

        # masks = pred_masks[0].squeeze().detach().cpu().numpy() > self.sam_lora.lora_vit.mask_threshold

        if file_name is not None:
            result = np.concatenate((pred_logits.detach().cpu().numpy() > self.sam_lora.lora_vit.mask_threshold,
                                     pred_logits_flip.detach().cpu().numpy()> self.sam_lora.lora_vit.mask_threshold),
                                    axis=-1)
            os.makedirs(os.path.join(self.cfg.work_dir, 'output_result'), exist_ok=True)
            cv2.imwrite(os.path.join(self.cfg.work_dir, 'output_result', '{}.png'.format(file_name)),
                        result[0].astype(np.uint8)*255)
        return masks

    def forward_step(self, data, optimizer, save=True, memory_bank=True):
        # 保证img gt bbox的shape都是[bs, xxx]  第一维是bs
        img, gt, bbox, file_path = data['img'], data['gt'], data['bboxes'], data['file'][0]  # bs=1
        file_name = file_path.split('/')[-1][:-4]
        self.forward_clip([file_path], optimizer[0])

        # get result. 这里注意如果只用positive text prompt效果是不是会好一些
        cat_img, saliency_map, img_np_uint = get_saliency_map(
            file_path, self.text_prompt[:], self.cam, self.preprocess, self.tokenizer, img_size=img.shape[-2:])
        crf_img = get_crf_map(img_np_uint.copy(), saliency_map)  # numpy format

        # memory_bank
        if memory_bank:
            crf_img_refine = self.refine_crf_map_memory_map(crf_img, img)
            # masks = get_sam_output(self.sam_predictor, img_np_uint, crf_img_refine)
            masks = self.forward_sam(img, crf_img_refine, optimizer[1], file_name=None)
        else:
            # masks = get_sam_output(self.sam_predictor, img_np_uint, crf_img)
            masks = self.forward_sam(img, crf_img, optimizer[1])
            crf_img_refine = None
        if save:
            save_path = os.path.join(self.cfg.work_dir, 'Output')
            os.makedirs(save_path, exist_ok=True)
            save_results(saliency_map, crf_img, masks, gt.squeeze().numpy(), save_path, file_name, crf_img_refine)
        return masks


