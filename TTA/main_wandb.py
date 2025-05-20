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
# wanbd setting
import wandb
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description='Test-time adaptation')
    parser.add_argument('--config', help='train config file path', default='configs/tta/tta_Fundus.py')
    parser.add_argument('--source', help='source domain', default='ORIGA')
    parser.add_argument('--domain', help='target domain', default='ORIGA')
    parser.add_argument('--baseline', help='baseline', default='SoueceOnly')
    parser.add_argument('--wandb', help='wandb config yaml', default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.source = args.source
    cfg.domain = args.domain
    cfg.baseline = args.baseline
    # work_dir = './Run/{}/{}'.format(args.baseline, args.domain)
    # os.makedirs(work_dir, exist_ok=True)
    # print('Save to: {}'.format(work_dir))
    # cfg.work_dir = work_dir

    project_name = args.baseline
    os.environ['WANDB_DIR'] = './Run/wandb/{}'.format(project_name)
    os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)

    with wandb.init() as run:

        wandb_cfg = run.config
        # 这里好像改了online才有效，这里就是控制每一个run的名字，但是offline应该是没用的
        run.name = "{}:lr{}_alpha{}_threshold{}_memorysize{}_neighborsize{}_lorablock{}".format(
            args.domain, wandb_cfg.learning_rate, wandb_cfg.alpha, wandb_cfg.threshold, wandb_cfg.memory_size,
            wandb_cfg.neighbor_size, wandb_cfg.lora_block)

        work_dir = './Run/{}/{}/lr{}_alpha{}_threshold{}_memorysize{}_neighborsize{}_lorablock{}'.format(
            args.baseline, args.domain, wandb_cfg.learning_rate, wandb_cfg.alpha, wandb_cfg.threshold,
            wandb_cfg.memory_size, wandb_cfg.neighbor_size, wandb_cfg.lora_block)
        os.makedirs(work_dir, exist_ok=True)
        print('Save to: {}'.format(work_dir))
        cfg.work_dir = work_dir

        cfg.learning_rate = wandb_cfg.learning_rate
        cfg.alpha = wandb_cfg.alpha
        cfg.threshold = wandb_cfg.threshold
        cfg.memory_size = wandb_cfg.memory_size
        cfg.neighbor_size = wandb_cfg.neighbor_size
        cfg.lora_block = wandb_cfg.lora_block

        dataset = build_dataset(cfg.dataset.root, cfg, cfg.domain, cfg.baseline)
        solver = build_solver(cfg.baseline)(cfg=cfg, dataset=dataset)
        solver.forward()


if __name__ == '__main__':
    worker_seed = 0
    worker_init_fn(0, 0, 0, seed=worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    args = parse_args()

    # wandb
    wandb_config = yaml.load(open(r'configs/wandb_sweep/{}.yaml'.format(args.wandb)), Loader=yaml.FullLoader)
    project_name = wandb_config['name']
    wandb_config['name'] = args.baseline + '_' + args.domain  # sweep name，在project下的name
    os.environ['WANDB_DIR'] = './Run/wandb/{}'.format(project_name)
    os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
    sweep_id = wandb.sweep(wandb_config, project='{}'.format(project_name))
    # print(sweep_id)
    wandb.agent(sweep_id, main, count=30)
