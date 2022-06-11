# -*- coding: utf-8 -*-

import os
from parser.cmds import Evaluate
import torch
from easydict import EasyDict as edict
import yaml
import click
import random
import numpy as np


@click.command()
@click.option("--eval_dep", default=False, help="evaluate dependency, only for N(B)L-PCFG")
@click.option("--data_split", default='test')
@click.option("--decode_type", default='mbr', help="viterbi or mbr")
@click.option("--load_from_dir", default="")
@click.option("--device", '-d', default='0')
def main(eval_dep, data_split, decode_type, load_from_dir, device):
    yaml_cfg = yaml.load(open(load_from_dir + "/config.yaml", 'r'))
    args = edict(yaml_cfg)
    args.device = device
    args.load_from_dir = load_from_dir
    print(f"Set the device with ID {args.device} visible")
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Set the random seed for reproducible experiments
    if hasattr(args, 'seed'):
        # Python
        random.seed(args.seed)
        # Numpy
        np.random.seed(args.seed)
        # Pytorch
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUDA
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if torch.version.cuda >= str(10.2):
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
            # or
            # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
        else:
            os.environ['CUDA_LAUNCH_BLOCKING']='1'

    command = Evaluate()
    command(args, data_split=data_split, decode_type=decode_type, eval_dep=eval_dep)


if __name__ == '__main__':
    main()