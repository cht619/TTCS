import os.path
import csv
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2
from PIL import Image
from .sam_utils import dice_coeff
from ..saliency_maps.clip_model import load_clip_cam
from ..segment_anything.build_model import build_sam_predictor
from .utils import process_img_clip, get_saliency_map, get_crf_map, get_sam_output, get_text_prompt, save_results
from ..models.coop_clip import ClipTestTimeTuning, ClipTestTimeTuning_VP
from ..models.sam import SAMModel


def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset

    Return:
        (list): list of batched dataset so a list(dict)
    """
    # for i in range(len(batch)):
    #     batch[i][-1] = 0  # 不在这里改了，因为这里改需要preprocess
    return list(batch)


def CHECK_NUM_PARAMS(model, lr=0.0):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Len params to train:{} and Lr is {}'.format(params, lr))
    return params


class SourceOnly:  # CLIP 不更新作为SO
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        self.clip_model, self.preprocess, self.tokenizer, self.cam = load_clip_cam()
        # self.clip_model.load_state_dict(torch.load('./REFUGE_Valid.pth'))
        self.cam.model = self.clip_model
        self.sam_predictor = build_sam_predictor()
        # self.text_prompt = 'An image of a fundus photographs for glaucoma assessment'
        self.text_prompt, labels = get_text_prompt(cfg.domain, cfg.baseline)  # 可以输多个prompt，所以这里仍然可以用来判断！！但是多个好像还不如1个某些dataset上
        self.labels = torch.tensor(labels).float().cuda()  # mse loss
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)  # 每次先处理bs=1看看

    def set_model_status(self):
        self.clip_model.eval()
        self.sam_predictor.model.eval()

    def forward_step(self, data, save=True):
        # 保证img gt bbox的shape都是[bs, xxx]  第一维是bs
        img, gt, bbox, file_path = data['img'], data['gt'], data['bboxes'], data['file'][0]  # bs=1
        file_name = file_path.split('/')[-1][:-4]
        cat_img, saliency_map, img_np_uint = get_saliency_map(
            file_path, self.text_prompt, self.cam, self.preprocess, self.tokenizer, img_size=img.shape[-2:])

        crf_img = get_crf_map(img_np_uint.copy(), saliency_map)
        masks = get_sam_output(self.sam_predictor, img_np_uint, crf_img)
        if save:
            save_path = os.path.join(self.cfg.work_dir, 'Output')
            os.makedirs(save_path, exist_ok=True)
            save_results(saliency_map, crf_img, masks, gt.squeeze().numpy(), save_path, file_name)
        return masks

    def forward(self):
        dice_scores = []
        with tqdm(total=len(self.dataloader), position=0) as pbar:
            for n, data in enumerate(self.dataloader):
                if data['gt'].shape[-2:] != data['img'].shape[-2:]:  # 为什么会有一个图片不一样
                    data['gt'] = F.interpolate(
                        data['gt'].unsqueeze(1).float(), data['img'].shape[2:], mode='bilinear').squeeze(1).long()

                preds = self.forward_step(data)
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


class CLIP_TTT:
    # refine clip to generate more useful prompt. 通过优化一个更好的prompt
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        clip_model, self.preprocess, self.tokenizer, self.cam = load_clip_cam()
        self.clip_model = clip_model.model
        self.clip_model.cuda()
        # self.text_prompt = 'An image of a fundus photographs for glaucoma assessment'
        self.text_prompt = self.tokenizer(text_prompt).cuda()
        self.model = ClipTestTimeTuning_VP(self.clip_model)
        self.model.cuda()
        # build dataset
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)


        # self.load_prompt_coop()  # 没有合适的权重

    def load_prompt_coop(self,
                         pth='./data/Pth/clip/coop/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'):
        # 这里因为官方并没有ViT-L/14的weight，这里的dim是768不是512，所以无办法利用pretrain weight
        print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(pth, map_location='cuda')['state_dict']['ctx']  # 只用获取ctx就行

        assert pretrained_ctx.size()[0] == 4
        self.model.prompt_learner.ctx.copy_(pretrained_ctx)  # 这里为啥有个0的
        self.model.prompt_learner.ctx_init_state = pretrained_ctx

    def build_optimizer(self, mode='prompt'):
        if mode == 'prompt':
            parameters = self.model.visual_prompt.parameters()

        optimizer = optim.Adam(parameters, lr=7e-3, weight_decay=5e-4)
        return optimizer

    def data_path_to_tensor(self, file_paths):
        data = torch.stack([self.preprocess(Image.open(file)) for file in file_paths])
        return data.cuda()

    def save_cam_img(self, img_path):
        self.cam.model.model = self.clip_model
        file_name = img_path.split('/')[-1][:-4]
        cat_img, saliency_map = get_saliency_map(img_path, text_prompt[0], self.cam, self.preprocess, self.tokenizer)
        # cat_img.save(os.path.join(self.cfg.work_dir, '{}.png'.format(file_name)))
        cv2.imwrite(os.path.join(self.cfg.work_dir, '{}.png'.format(file_name)), saliency_map)

    def forward_step(self, data, optimizer):
        labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]).float().cuda()
        # 这里更新好像有好几种方式，可以直接用prompt
        _, gt, file_path = data
        image = self.data_path_to_tensor(file_path)
        image_logits, text_logits = self.model(image, self.text_prompt)
        loss = F.binary_cross_entropy_with_logits(image_logits, labels.unsqueeze(0)) / len(text_prompt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # save result
        self.save_cam_img(img_path=file_path[0])

        if max(image_logits)[1].item() < 34:
            self.clip_model.visual_prompt.reset_prompt()
        return max(image_logits)[1].item()

    def forward(self):
        optimizer = self.build_optimizer()
        with tqdm(total=len(self.dataloader), position=0) as pbar:
            for n, data in enumerate(self.dataloader):
                scores = self.forward_step(data, optimizer)
                pbar.update(1)
                pbar.set_description('[Process:{}. Clip scores:{:.2f}]'.format(
                    data[-1][0].split('/')[-1], scores))

        torch.save(self.model.state_dict(), './{}.pth'.format(self.cfg.domain))


class SAM_Prompt_exp(SourceOnly):
    '''
    如果直接用gt prompt，结果怎么样？
    给定前几个prompt，结果会有明显提升吗？
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SAMModel(self.sam_predictor.model).cuda()

    def build_optimizer(self):
        pass

    @torch.no_grad()
    def forward_step(self, data, optimizer=None, save=True):
        img, gt, bbox, file_path = data['img'], data['gt'], data['bboxes'], data['file'][0]  # bs=1
        _, pred_masks, _, _ = self.model(img.cuda(), bbox.cuda())


        # loss = self.seg_cost(stk_out, stk_gt.float().cuda())
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        if save:
            save_path = os.path.join(self.cfg.work_dir, 'Output')
            os.makedirs(save_path, exist_ok=True)
            result = pred_masks[0].squeeze().detach().cpu().numpy()
            result = np.stack((result, result, result), axis=-1)
            gt = gt.squeeze().detach().cpu().numpy()
            gt = np.stack((gt, gt, gt), axis=-1)

            img = img.detach().squeeze().cpu().permute(1, 2, 0).numpy()
            bbox_ = bbox.squeeze().numpy().astype(np.uint8)
            img[bbox_[0]: bbox_[0] + bbox_[2], bbox_[1]: bbox_[1] + bbox_[3], :] = [255, 0, 0]
            img = np.hstack((img, result*255, gt*255))
            file_name = file_path.split('/')[-1][:-4]
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(file_name)), img)

        return pred_masks[0].squeeze().detach().cpu().numpy()