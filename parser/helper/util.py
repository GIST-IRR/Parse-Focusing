import time
import os
import logging
from distutils.dir_util import copy_tree

from parser.model import NeuralPCFG, CompoundPCFG, TNPCFG, NeuralBLPCFG, NeuralLPCFG

import torch


def get_model(args, device='cpu'):
    if args.model_name == 'NPCFG':
        return NeuralPCFG(args).to(device)

    elif args.model_name == 'CPCFG':
        return CompoundPCFG(args).to(device)

    elif args.model_name == 'TNPCFG':
        return TNPCFG(args).to(device)

    elif args.model_name == 'NLPCFG':
        return NeuralLPCFG(args).to(device)

    elif args.model_name == 'NBLPCFG':
        return NeuralBLPCFG(args).to(device)

    else:
        raise KeyError


def get_optimizer(args, model):
    if args.name == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu))
    elif args.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

def get_logger(args, log_name='train',path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = "/{}".format(model_name) \
        + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return  saved_name

