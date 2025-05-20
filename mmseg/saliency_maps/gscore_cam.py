import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
import cv2
from .utils.svd_on_activations import get_2d_projection
from .utils.image import scale_cam_image, normalize2D, resize2D, cls_reshpae


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []

        return self.model(*x) if isinstance(x, tuple) else self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = True,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 is_clip: bool = False,
                 is_transformer: bool = False,
                 model_domain: str = None,
                 **kwargs,
                 ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers,
            reshape_transform=None)  # ! set reshape_transform to None, because we will use the reshape_transform of inside BaseCAM to aviod unexpected function call of reshape_transform

        allowed_keys = {'drop', 'topk', 'img_size', 'channels', 'groups'}
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.is_clip = is_clip
        self.is_transformer = is_transformer
        self.model_domain = model_domain

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        # targets: List[torch.nn.Module],
                        targets: np.array,
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise NotImplementedError("Base cam should not be used by itself.")

    def dim_mapper(self, activation_shape: tuple, weight_shape: tuple) -> str:
        dim_s = 'ncwh'
        weight_length = len(weight_shape)
        start_dim = activation_shape.index(weight_shape[0])
        return dim_s[start_dim: start_dim + weight_length]

    def getRawActivation(self, input_tensor, target_category=None, img_size=None):
        if self.is_clip:
            output = self.activations_and_grads(input_tensor)[1]  # per text logit of clip
        else:
            output = self.activations_and_grads(input_tensor)

        # self.model.zero_grad()
        # output.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        # grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = activations[0]

        cam = np.maximum(cam, 0)

        result = []
        # fix bug in cv2 that it does not support type 23. (float16)
        if cam.dtype == 'float16':
            cam = cam.astype(np.float32)
        input_shape = img_size if self.is_clip else input_tensor.shape[-2:][::-1]
        for img in cam:
            img = cv2.resize(img, input_shape)
            img = img - np.min(img)
            img = img / (np.max(img) + 1e-8)
            result.append(img)
        result = np.float32(result)
        return result

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      #   targets: List[torch.nn.Module],
                      targets: np.array,
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        self.weights = self.get_cam_weights(input_tensor,
                                            target_layer,
                                            targets,
                                            activations,
                                            grads)  # usually (n, c), but with einsum, now is support any shapes that match the corresponding dimension of activation. i.e., (c, w, h)
        # weighted_activations = weights[:, :, None, None] * activations
        weight_dim = self.dim_mapper(activations.shape, self.weights.shape)
        weighted_activations = np.einsum(f"{weight_dim},ncwh->ncwh", self.weights.astype(np.float32),
                                         activations.astype(np.float32))

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                # targets: List[torch.nn.Module],
                targets: np.array,
                eigen_smooth: bool = False) -> np.ndarray:

        if self.is_clip:
            if self.cuda:
                input_tensor = (input_tensor[0].cuda(), input_tensor[1].cuda())

            if self.compute_input_gradient:
                input_tensor = (torch.autograd.Variable(input_tensor[i],
                                                        requires_grad=True)
                                for i in len(input_tensor))
            outputs = self.activations_and_grads(input_tensor)[1]  # per text logit of clip
        else:
            if self.cuda:
                input_tensor = input_tensor.cuda()

            if self.compute_input_gradient:
                input_tensor = torch.autograd.Variable(input_tensor,
                                                       requires_grad=True)

            outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            # targets = [ClassifierOutputTarget(category) for category in target_categories]
            targets = target_categories
        else:
            target_categories = targets

        if self.uses_gradients:
            self.model.zero_grad()
            # loss = sum(target(output) for target, output in zip(targets, outputs))
            loss = sum(outputs[:, target_categories])
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            # targets: List[torch.nn.Module],
            targets: np.array,
            eigen_smooth: bool) -> np.ndarray:
        if (self.model_domain == "biomedclip"):
            for a in range(len(self.activations_and_grads.activations)):
                self.activations_and_grads.activations[a] = self.activations_and_grads.activations[a].permute(1, 0, 2)
        activations_list = [self.reshape_transform(a).numpy() if self.reshape_transform is not None else a.numpy()
                            for a in self.activations_and_grads.activations]
        self.activations = activations_list
        if self.is_transformer:  # use the gradient of CLS token for gradient
            grads_list = [cls_reshpae(g) for g in self.activations_and_grads.gradients]
        else:
            grads_list = [self.reshape_transform(g).numpy() if self.reshape_transform is not None else g.numpy()
                          for g in self.activations_and_grads.gradients]

        if hasattr(self, 'img_size'):
            target_size = self.img_size
        else:
            target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam,
                             0)  # * The difference between having this line is small, i.e., in COCO val, with this line is 20.83% and without this line is 20.86%
            # normalize and resize cam
            cam = np.array([normalize2D(img) for img in cam])
            if target_size is not None:  # * no resize such that we can post process it later
                cam = np.array([resize2D(img, target_size) for img in cam])
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        # return scale_cam_image(result)
        return np.array([normalize2D(cam) for cam in result])

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       #    targets: List[torch.nn.Module],
                                       targets: np.array,
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class GScoreCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=True, reshape_transform=None, is_clip=False, drop=False, mute=True, topk=None, channels=None, batch_size: int = 128, is_transformer: bool = False,
                 model_domain=None):
        super(GScoreCAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform=reshape_transform, is_clip=is_clip, drop=drop, mute=mute, batch_size=batch_size, is_transformer=is_transformer)
        self.topk = topk
        self.use_bot  = False
        self.drop = drop
        self.mute = mute
        self.model_domain = model_domain

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads
                        ):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # with open('context_text.json', 'r'):
            if self.is_clip:
                # img_size = img_size
                img_tensor, text_tensor = input_tensor[0], input_tensor[1]
            else:
                img_tensor = input_tensor
            img_size = img_tensor.shape[-2 : ]
            upsample = torch.nn.UpsamplingBilinear2d(img_size)
            activation_tensor = torch.from_numpy(activations)  # 原来的shape is [197, 1, 768]，需要经过reshape
            # if self.cuda:
            #     activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor.float())

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins +1e-6)  # NOTE this could lead to division by zero

            input_tensors = img_tensor[:, None, :, :].cpu() * upsampled[:, :, None, :, :]

            BATCH_SIZE = self.batch_size if hasattr(self, "batch_size") else 64
            k = 300 if self.topk is None else self.topk
            # * testing for different vairance
            # ^ average gradient
            importance = torch.from_numpy(grads).float().mean(axis=(2, 3))
            # ^ max pooling
            # maxpool = torch.nn.MaxPool2d(grads.shape[2:])
            # importance = maxpool(torch.from_numpy(abs(grads)).float()).view((1, -1))
            # ^ average pooling
            # averagepool = torch.nn.AvgPool2d(grads.shape[2:])
            # importance = averagepool(torch.from_numpy(abs(grads)).float()).view((1, -1))

            if self.use_bot:
                indices_top = importance.topk(k)[1][0]
                indices_bot = importance.topk(k, largest=False)[1][0]
                indices = torch.cat([indices_top, indices_bot])
            else:
                indices = importance.topk(k)[1][0]

            scores = []
            top_tensors = input_tensors[:, indices]
            if isinstance(target_category, int):
                target_category = [target_category]

            # 应该是先Test Time Augmentation，然后再计算gradient
            for category, tensor in zip(target_category, top_tensors):
                # for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE), disable=self.mute):
                for i in range(0, tensor.size(0), BATCH_SIZE):

                    batch = tensor[i: i + BATCH_SIZE, :]
                    if self.is_clip:
                        outputs = self.model(batch.cuda(), text_tensor.cuda())[0].cpu().numpy()[:, category]
                    else:
                        outputs = self.model(batch.cuda()).cpu().numpy()[:, category]
                    scores.extend(outputs)

            scores = torch.Tensor(scores)
            if scores.isnan().any():
                scores = scores.nan_to_num(nan=0.0)  # fix nan bug in clip implementation

            # place the chosen scores back to the weight
            emtpy_score = torch.zeros(activations.shape[1])
            emtpy_score[indices] = scores
            scores = emtpy_score.view(activations.shape[0], activations.shape[1])

            weights = torch.nn.Softmax(dim=-1)(scores)
            if self.use_bot:
                bot_mask = torch.ones(activations.shape[1])
                bot_mask[indices_bot] = -1
                bot_mask = bot_mask.view(activations.shape[0], activations.shape[1])
                weights = weights * bot_mask
            return weights.numpy()


class CAMWrapper(object):
    CAM_LIST = ['gscorecam']
    CAM_DICT = {
                "gscorecam": GScoreCAM,  # default
                }

    def __init__(self, model, target_layers, tokenizer, cam_version, preprocess=None, target_category=None,
                 is_clip=True,
                 mute=False, cam_trans=None, is_transformer=False, model_domain="biomedclip", **kwargs):
        """[summary]

        Args:
            model (model): [description]
            target_layers (model layer): List[layers]
            drop (bool, optional): [description]. Defaults to False.
            cam_version (str, optional): [description]. Defaults to 'gradcam'.
            target_category (int or tensor, optional): [description]. Defaults to None.
            mute (bool, optional): [description]. Defaults to False.
            channel_frame (csv, optional): [description]. Defaults to None.
            channels (int, optional): [description]. Defaults to None.
            cam_trans (function, optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        self.mute = mute
        self.model = model
        self.version = cam_version
        self.target_layers = target_layers
        self.target_category = target_category
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.cam_trans = cam_trans
        self.is_transformer = is_transformer
        self.is_clip = is_clip
        self.channels = None
        self.__dict__.update(kwargs)
        self.model_domain = model_domain

        if self.version not in self.CAM_LIST:
            raise ValueError(f"CAM version not found. Please choose from: {self.CAM_LIST}")
        # define cam
        self._load_cam()

    def _select_channels(self, text):
        if self.channel_dict is not None and text in self.channel_dict.keys():
            return self.channel_dict[text][:self.topk]
        else:
            return None

    def _load_channel_from_csv(self, channel_frame):  # !deprecated
        import pandas as pd
        channelFrame = pd.read_csv(channel_frame, index_col=0)
        return channelFrame[channelFrame.columns[0]].to_list()

    # load cam
    def _load_cam(self):
        if self.version in ['scorecam', 'gscorecam']:
            batch_size = self.batch_size if hasattr(self, "batch_size") else 128
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers,
                                                   use_cuda=True, is_clip=self.is_clip,
                                                   reshape_transform=self.cam_trans, drop=self.drop,
                                                   mute=self.mute, channels=self.channels, topk=self.topk,
                                                   batch_size=batch_size, is_transformer=self.is_transformer,
                                                   model_domain=self.model_domain)

    def getCAM(self, input_img, input_text, cam_size, target_category, model_domain):
        cam_input = (input_img, input_text) if self.is_clip else input_img
        self.cam.img_size = cam_size
        if self.version == 'hilacam':
            grayscale_cam = self.cam(input_img, input_text, self.model, 'cuda', cam_size=cam_size,
                                     index=target_category)
        elif self.version == 'groupcam':
            grayscale_cam = self.cam(cam_input, class_idx=target_category)
            grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0)
        elif self.version.startswith('sscam'):
            grayscale_cam = self.cam(input_img, input_text, class_idx=target_category,
                                     param_n=35, mean=0, sigma=2, mute=self.mute)
        elif self.version == 'layercam':
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        elif self.version == 'rise':
            grayscale_cam = self.cam(inputs=cam_input, targets=target_category, image_size=cam_size)
        # elif self.version == 'lime':
        #     grayscale_cam = self.cam(inputs=cam_input, target=target_category, image_size=(224, 224), image=kwargs['image'])
        else:
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def __call__(self, inputs, label, heatmap_size, model_domain):

        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
        else:
            img = inputs
            text = None

        if self.preprocess is not None:
            img = self.preprocess(img)
        # tokenize text
        text_token = None if self.tokenizer is None else self.tokenizer(text).cuda()
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        if not img.is_cuda:
            img = img.cuda()
        if hasattr(self, "channel_dict"):
            # self.cam.channels = self.channel_dict[text]
            self.cam.channels = self._select_channels(text)

        return self.getCAM(img, text_token, heatmap_size, label, model_domain)

    def getLogits(self, img, text):
        with torch.no_grad():
            if self.preprocess is not None:
                img = self.preprocess(img)
            img_per_text, text_per_img = self.model(img.unsqueeze(0).cuda(), self.tokenizer(text).cuda())
        return img_per_text, text_per_img