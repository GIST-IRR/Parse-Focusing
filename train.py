# -*- coding: utf-8 -*-

import argparse
from time import sleep
from parser.cmds import Evaluate, Train
from pathlib import Path


from torch_support import reproducibility as reprod
from torch_support.train_support import (
    setup_log_dir,
    get_config_from,
    command_decorator
)
from torch_support.device_support import set_device

import copy

def train(args2, device=None):
    if device is not None:
        args2.device = device
    # load config
    args = get_config_from(args2.conf)
    args.update(args2.__dict__)

    # update save directory
    config_path = Path(args2.conf)
    config_name = config_path.stem
    args.save_dir = args.save_dir + f"/{config_name}"
    
    # generate save directory
    setup_log_dir(args.save_dir)

    # set device
    args.device = set_device(args.device)

    # train
    train_model = command_decorator(Train())
    train_model(args)

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