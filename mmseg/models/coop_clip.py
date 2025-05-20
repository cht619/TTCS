import torch
from torch import nn
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
from clip import load
from typing import List, Tuple


# clip_model, _ = load('ViT-B/16')  # 可以用来看一些信息
# print(dir(clip_model))


_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text  # 这里就是text
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        # 从x张量中选择出每个样本对应的最可能token的文本编码结果
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end',
                 learned_cls=False, my_tokenizer=None):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = torch.float32  # bio clip是32，其他是16
        self.dtype = torch.float32
        self.device = torch.device('cuda')
        # ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_dim = 768  # 这里对应的是text prompt after tokenizer后的dim，一般都是768

        # ctx_vectors.shape should be [n_ctx, ctx_dim]

        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        if my_tokenizer is not None:
            self.tokenizer = my_tokenizer
        else:
            self.tokenizer = tokenize

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                # embedding = clip_model.token_embedding(prompt).type(dtype)
                embedding = clip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)

            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 这里只能自动更新适应的，因为没有合适的checkpoint
            # this. Random initialization. then load from pretrain pth
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.prompt_prefix = prompt_prefix  # learnable prompt

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if not self.learned_cls:  # this
            classnames = [name.replace("_", " ") for name in classnames]  # a photo of class name
            # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # prompts = [prompt_prefix + " " + name + "." for name in classnames]
            prompts = [prompt_prefix + " " + 'An image of a fundus photographs for glaucoma assessment' + "."]  # one prompt for one domain
            name_lens = [len(_tokenizer.encode('An image of a fundus photographs for glaucoma assessment'))]
            print('the prompt is :', prompts, _tokenizer.encode('An image of a fundus photographs for glaucoma assessment'))
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        # 这里注意tokenizer要永辉bio_clip的，不然无法正常计算，维度对不上
        # 这个tokenized_prompts就是原来的prompt
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            # embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            # 则合理的embedding，关于给定的prompt就是固定的。只需要学习前面几个lenable prompt
            embedding = clip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            print('the prompt embedding shap is: ', embedding.shape)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx  # 更新ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:  # True
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
            prompts = [self.prompt_prefix + " " + 'An image of a fundus photographs for glaucoma assessment' + "."]  # one prompt for one domain
            name_lens = [len(_tokenizer.encode('An image of a fundus photographs for glaucoma assessment'))]

        else:
            # assume each learnable cls_token is only 1 word
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            # prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            prompts = [self.prompt_prefix + " " + 'An image of a fundus photographs for glaucoma assessment' + "."]  # one prompt for one domain
            name_lens = [len(_tokenizer.encode('An image of a fundus photographs for glaucoma assessment'))]

            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None, batch_size=1):
        self.batch_size = None
        # the init will be used when computing CLIP directional loss
        ctx = self.ctx  # [n_ctx, ctx_dim]
        ctx = ctx.unsqueeze(0).expand(1, -1, -1)  # 1个domain只有1个class，所以self.n_cls=1
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # elif not ctx.size()[0] == self.n_cls:
        #     ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix  # [1, 1, 768]
        suffix = self.token_suffix  # [1, 251, 768]  # 中间刚好是prompt，即n_ctx的数目

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=-2,
        )

        return prompts  # 和embeddings的shape一样，比如[1, 256, 768]


# coop or cocoop
# coop
class ClipTestTimeTuning(nn.Module):
    def __init__(self, model, prompts, batch_size=None, criterion='cosine', arch="ViT-L/14",
                 n_ctx=4, ctx_init=None, ctx_position='end', learned_cls=False, tokenizer=None):
        super(ClipTestTimeTuning, self).__init__()
        self.model = model
        self.image_encoder = model.visual
        # self.text_encoder = TextEncoder(model)
        self.text_encoder = model.text
        self.logit_scale = model.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(model, prompts, batch_size, n_ctx, ctx_init, ctx_position, learned_cls,
                                            my_tokenizer=tokenizer)
        self.criterion = criterion
        self.dtype = torch.float32

    def forward(self, image):
        # 只更新prompt
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        print(image_features.shape)
        text_features = self.get_text_features(batch_size=image.shape[0])
        print(text_features.shape)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_text_features(self, batch_size):
        text_features = []
        prompts = self.prompt_learner.forward(batch_size=batch_size)
        print(prompts.shape)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.forward_text_features(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)


class VisualPrompt(nn.Module):
    def __init__(self, prompt_alpha=0.5, image_size=512):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

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
        prompt = F.pad(self.data_prompt,
                       [self.padding_size, imgH - self.padding_size - self.prompt_size, self.padding_size,
                        imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_

    @torch.no_grad()
    def reset_prompt(self):
        reset_data = torch.ones_like(self.data_prompt).to(self.data_prompt.device)
        self.data_prompt.copy_(reset_data)


# 直接更新visual prompt. 应该会比text prompt更好（没有checkpoint，要修改的地方太多）
class ClipTestTimeTuning_VP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.visual_prompt = VisualPrompt(image_size=224)

    def forward(self, image, text):
        prompt_x, low_x = self.visual_prompt(image)
        image_features = self.model.encode_image(prompt_x, normalize=True) if prompt_x is not None else None
        with torch.no_grad():
            text_features = self.model.encode_text(text, normalize=True) if text is not None else None
        out_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.model.logit_scale.exp()
        }
        if self.model.logit_bias is not None:
            out_dict['logit_bias'] = self.model.logit_bias
        image_logits = (self.model.logit_scale.exp() * image_features @ text_features.t())
        text_logits = (self.model.logit_scale.exp() * text_features @ image_features.t())  # is a score
        return image_logits, text_logits  # 其实这两个是一样的
