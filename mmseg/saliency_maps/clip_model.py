import torch
from torch import nn
import os
from clip import load
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from .gscore_cam import CAMWrapper


def reshape_transform(tensor, height=None, width=None):
    # 这里用在CAM的reshape，必须有
    if height or width is None:
        grid_square = len(tensor) - 1
        if grid_square ** 0.5 % 1 == 0:
            height = width = int(grid_square ** 0.5)
        else:
            raise ValueError("Heatmap is not square, please set height and width.")
    result = tensor[1:, :, :].reshape(
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(2, 0, 1)
    return result.unsqueeze(0)


class BiomedCLIP(nn.Module):
    def __init__(
            self,
            model
    ):
        super().__init__()
        self.model = model

    def forward(self, images, texts):
        # Getting Image and Text Features

        image_embeddings, text_embeddings, logit_scale = self.model(images, texts)

        # Calculating the Loss
        image_logits = (logit_scale * image_embeddings @ text_embeddings.t())
        text_logits = (logit_scale * text_embeddings @ image_embeddings.t())  # is a score

        return image_logits, text_logits  # 这里就是多了个text_logits


class VisualPrompt(nn.Module):
    def __init__(self, prompt_alpha=0.5, image_size=512):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data
        self.image_size = image_size

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone().float(), dim=(-2, -1))
        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
        if imgH == self.image_size:  # 和原始prompt即clip的输入一致
            prompt = F.pad(self.data_prompt,
                           [self.padding_size, imgH - self.padding_size - self.prompt_size, self.padding_size,
                            imgW - self.padding_size - self.prompt_size],
                           mode='constant', value=1.0).contiguous()
        else:
            prompt = F.interpolate(self.data_prompt, size=(imgH, imgW))
        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_

    @torch.no_grad()
    def reset_prompt(self):
        reset_data = torch.ones_like(self.data_prompt).to(self.data_prompt.device)
        self.data_prompt.copy_(reset_data)


class BiomedCLIP_VP(BiomedCLIP):
    def __init__(self, model):
        super().__init__(model=model)
        self.visual_prompt = VisualPrompt(image_size=224)

    def forward(self, images, texts):
        # Getting Image and Text Features
        images, _ = self.visual_prompt(images)
        image_embeddings, text_embeddings, logit_scale = self.model(images, texts)

        # Calculating the Loss
        image_logits = (logit_scale * image_embeddings @ text_embeddings.t())
        text_logits = (logit_scale * text_embeddings @ image_embeddings.t())  # is a score

        return image_logits, text_logits  # 这里就是多了个text_logits


def load_clip_cam(clip='ViT-B/16', model_path='./data/Pth', cam_versions='gscorecam', cache_dir='./data/Pth',
                  use_visual_prompt=False):
    # target_layer =
    # _, _, _, cam_trans, clip = load_clip(self.clip_version, resize=self.resize, custom=self.custom_clip)

    # preprocess: Compose(
    #     Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    #     CenterCrop(size=(224, 224))
    #     <function _convert_to_rgb at 0x7f167d3741f0>
    #     ToTensor()
    #     Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )
    biomedclip_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir=cache_dir)
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    if not use_visual_prompt:
        clip_model = BiomedCLIP(biomedclip_model)
    else:
        clip_model = BiomedCLIP_VP(biomedclip_model)
    # clip_model.visual.transformer.resblocks[-1]
    target_layer = clip_model.model.visual.trunk.blocks[11].norm2
    cam = CAMWrapper(model=clip_model, target_layers=[target_layer], topk=300, drop=True, channels=None,
                     batch_size=128, model_domain='biomedclip', cam_version=cam_versions, tokenizer=tokenizer,
                     preprocess=preprocess, is_transformer=True, cam_trans=reshape_transform)  # 别忘了reshape_transform
    return clip_model, preprocess, tokenizer, cam
