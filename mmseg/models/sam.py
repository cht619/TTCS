import torch
from torch import nn, optim
import torch.nn.functional as F


class SAMModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, prompts):
        if len(prompts.shape) == 2:
            prompts = prompts.unsqueeze(0)  # should be [bs, n, 4]
        image_embeddings = self.encode(images)  # image embeddings
        pred_masks, ious, res_masks = self.decode(prompts)
        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        self.image_embeddings = self.model.image_encoder(images)  # 好像没有normalization，可能影响不大
        return self.image_embeddings

    def decode(self, prompts, image_embeddings=None):
        if image_embeddings is None:
            image_embeddings = self.image_embeddings

        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.to(device=embedding.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=prompt,  # bbox
                masks=None,
            )
            elif isinstance(prompt, tuple):  # data points
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=prompt,
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
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

            masks = masks > self.model.mask_threshold  # 别忙了用threshold判断

            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks  # 这里注意返回的是list，但是我们一般用第一个就i选哪个