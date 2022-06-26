# -*- coding: utf-8 -*-

import argparse
import os
from time import sleep
from parser.cmds import Evaluate, Train
import shutil
import torch
import traceback
from pathlib import Path
from easydict import EasyDict as edict
import yaml

from torch_support.train_support import fix_seed, setup_log_dir

import random
import numpy as np
import copy

def train(args2):
    yaml_cfg = yaml.load(open(args2.conf, 'r'))
    args = edict(yaml_cfg)
    args.update(args2.__dict__)

    print(f"Set the device with ID {args.device} visible")
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.device)

    config_path = Path(args.conf if args.conf else args2.load_from_dir + "/config.yaml")
    config_name = config_path.stem
    args.save_dir = args.save_dir + "/{}".format(config_name)

    if hasattr(args, 'seed'):
        fix_seed(args.seed)

    # Auto-generation of log dir
    log_dir = Path(args2.conf).stem
    setup_log_dir(f'log/{log_dir}')

    try:
        command = Train()
        command(args)
    except KeyboardInterrupt:
        command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
        if command == 0:
            shutil.rmtree(args.save_dir)
            print("You have successfully delete the created log directory.")
        else:
            print("log directory have been saved.")
    except Exception:
        traceback.print_exc()
        shutil.rmtree(args.save_dir)
        print("log directory have been deleted.")

def train_manager(args, event):
    event.set()
    train(args)
    event.clear()

def multi_train(args):
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    from multiprocessing import Pool, Manager
    n_device = args.n_device
    n_models = args.n_models
    events = [Manager().Event() for _ in range(n_device)]

    pool = Pool(processes=4)
    for i in range(n_models):
        print(f'process {i}')
        n = i % n_device
        event = events[n]
        targs = copy.copy(args)
        targs.device = str(n)
        pool.apply_async(train_manager, args=(targs, event,))
        sleep(5) # to avoid conflict between processes
        while all([e.is_set() for e in events]):
            sleep(1)

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PCFGs'
    )
    parser.add_argument('--conf', '-c', default='')
    parser.add_argument('--n_device', '-nd', type=int, default=1)
    parser.add_argument('--n_models', '-nm', type=int, default=1)
    parser.add_argument('--device', '-d', default='0')
    args = parser.parse_args()

    if args.n_device < 2:
        train(args)
    else:
        multi_train(args)