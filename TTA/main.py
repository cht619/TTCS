import argparse
import numpy as np
import random
from mmengine.dataset import worker_init_fn
import os
import sys  # 记得每次都要把路径加上，不要import同名的module有问题
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
current_path = os.path.dirname(__file__)
sys.path.append(current_path)
from mmseg.segment_anything import build_sam


import torch
from mmengine.config import Config
from TTA.main_train import build_dataset, build_solver



def parse_args():
    parser = argparse.ArgumentParser(description='Test-time adaptation')
    parser.add_argument('--config', help='train config file path', default='configs/tta/tta_Fundus.py')
    parser.add_argument('--source', help='source domain', default='ORIGA')
    parser.add_argument('--domain', help='target domain', default='ORIGA')
    parser.add_argument('--baseline', help='baseline', default='SoueceOnly')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.source = args.source
    cfg.domain = args.domain
    cfg.baseline = args.baseline
    work_dir = './Run/{}/{}'.format(args.baseline, args.domain)
    os.makedirs(work_dir, exist_ok=True)
    print('Save to: {}'.format(work_dir))
    cfg.work_dir = work_dir
    dataset = build_dataset(cfg.dataset.root, cfg, cfg.domain, cfg.baseline)
    solver = build_solver(cfg.baseline)(cfg=cfg, dataset=dataset)
    solver.forward()


if __name__ == '__main__':
    worker_seed = 0
    worker_init_fn(0, 0, 0, seed=worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    main()