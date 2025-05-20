import os
from .build_sam import sam_model_registry
from .predictor import SamPredictor


model_pth_all = {'vit_h': 'sam_vit_h_4b8939.pth',
             'vit_l': 'sam_vit_l_0b3195.pth',
             'vit_b': 'sam_vit_b_01ec64.pth'}

def build_sam_predictor(model_type='vit_h', model_path='./data/Pth/SAM'):
    # 默认是ViT_h，可以跑三个backbone看看效果
    sam = sam_model_registry[model_type](checkpoint=os.path.join(model_path, model_pth_all[model_type]))
    _ = sam.cuda()
    predictor = SamPredictor(sam)
    return predictor