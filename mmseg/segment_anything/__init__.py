# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import sys

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 必需写init否则无法正常加载。好像也不能正常加载
# 这里的注释不能写，写了好像有问题，不能相互调用

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator
from .utils.transforms import ResizeLongestSide






