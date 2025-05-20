import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from .modeling import Sam



class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # x: [25, 14, 14, 768]; self.qkv: Linear(in_features=768, out_features=2304, bias=True)
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class LoRA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            # nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            nn.init.constant_(w_A.weight, 1)  # 01初始化是不是对结果影响最小?
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class LoRA_Sam(LoRA):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None, lora_block=3):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))  # 一共有31个block
            print(self.lora_layer)
            self.lora_layer = self.lora_layer[:lora_block]
            print(self.lora_layer, '---use which layers')
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False   # 更新哪个部分？
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)  # 更新这些地方
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        # self.sam = sam_model
        self.lora_vit = sam_model

    def encode(self, images):
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        self.image_embeddings = self.lora_vit.image_encoder(images)  # 好像没有normalization，可能影响不大
        return self.image_embeddings

    def decode(self, prompts, image_embeddings=None):
        if image_embeddings is None:
            image_embeddings = self.image_embeddings

        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):
            # if isinstance(prompt, torch.Tensor):
            #     prompt = prompt.to(device=embedding.device)
            #     sparse_embeddings, dense_embeddings = self.lora_vit.prompt_encoder(
            #     points=None,
            #     boxes=prompt,  # bbox
            #     masks=None,
            # )
            # elif isinstance(prompt, tuple):  # data points
            #     sparse_embeddings, dense_embeddings = self.lora_vit.prompt_encoder(
            #     points=prompt,
            #     boxes=None,
            #     masks=None,
            # )

            prompt = prompt.to(device=embedding.device)
            sparse_embeddings, dense_embeddings = self.lora_vit.prompt_encoder(
                points=None,
                boxes=prompt,  # bbox
                masks=None)

            low_res_masks, iou_predictions = self.lora_vit.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.lora_vit.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                self.image_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks

    def forward(self, img, prompt):
        image_embeddings = self.encode(img)  # image embeddings
        pred_masks, ious, res_masks = self.decode(prompt)
        #
        # pred_masks 是 res_masks interpolate的结果，即做了个resize
        return image_embeddings, pred_masks, ious, res_masks


class SAM_Model(nn.Module):
    def __init__(self, model):
        super().__init__()

        # lets freeze first
        for param in model.image_encoder.parameters():
            param.requires_grad = False   # 更新哪个部分？
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in model.mask_decoder.parameters():
            param.requires_grad = False

        for n, param in model.image_encoder.named_parameters():
            if 'attn' in n:
                param.requires_grad = True

        self.lora_vit = model


    def encode(self, images):
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        self.image_embeddings = self.lora_vit.image_encoder(images)  # 好像没有normalization，可能影响不大
        return self.image_embeddings

    def decode(self, prompts, image_embeddings=None):
        if image_embeddings is None:
            image_embeddings = self.image_embeddings

        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):
            # if isinstance(prompt, torch.Tensor):
            #     prompt = prompt.to(device=embedding.device)
            #     sparse_embeddings, dense_embeddings = self.lora_vit.prompt_encoder(
            #     points=None,
            #     boxes=prompt,  # bbox
            #     masks=None,
            # )
            # elif isinstance(prompt, tuple):  # data points
            #     sparse_embeddings, dense_embeddings = self.lora_vit.prompt_encoder(
            #     points=prompt,
            #     boxes=None,
            #     masks=None,
            # )

            prompt = prompt.to(device=embedding.device)
            sparse_embeddings, dense_embeddings = self.lora_vit.prompt_encoder(
                points=None,
                boxes=prompt,  # bbox
                masks=None)

            low_res_masks, iou_predictions = self.lora_vit.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.lora_vit.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                self.image_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks

    def forward(self, img, prompt):
        image_embeddings = self.encode(img)  # image embeddings
        pred_masks, ious, res_masks = self.decode(prompt)
        #
        # pred_masks 是 res_masks interpolate的结果，即做了个resize
        return image_embeddings, pred_masks, ious, res_masks