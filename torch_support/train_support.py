import os
import time
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree
import traceback
import logging

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

import random
import numpy as np
import torch

def get_config_from(path, verbose=False):
    if not isinstance(path, Path):
        path = Path(path)
    
    with path.open('r') as f:
        args = load(f, Loader=Loader)
    args = EasyDict(args)
    args.update({'conf': str(path)})

    if verbose:
        print(args)
    
    return args

def setup_log_dir(path, parents=True, exist_ok=True):
    if isinstance(path, (list, tuple)):
        path = [Path(p) for p in path]
    else:
        path = [Path(path)]

    for p in path:
        if not p.exists():
            p.mkdir(parents=parents, exist_ok=exist_ok)
            print(f'log dir [{p}] is created.')

def fix_seed(
        seed,
        python_seed=True,
        numpy_seed=True,
        torch_seed=True,
        cuda_seed=True,
        worker_init_fn=False,
        generator=False,
    ):
    '''
    If you want to fix seed,
    you have to call this function before assigning variables that you want to fix.
    '''
    # Python
    if python_seed:
        random.seed(seed)
    # Numpy
    if numpy_seed:
        np.random.seed(seed)
    # if you turn off funcs below, model is random, loader is fixed.
    # Pytorch
    if torch_seed:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # CUDA
    if cuda_seed:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.version.cuda >= str(10.2):
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
            # or
            # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
        else:
            os.environ['CUDA_LAUNCH_BLOCKING']='1'

    if worker_init_fn:
        def seed_worker(worker_id):
            worker_seed = seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        worker_init_fn = seed_worker
    
    if generator:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return worker_init_fn, generator

def create_save_path(args):
    model_name = args.model.model_name
    suffix = f'/{model_name}' \
        + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return  saved_name

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

def command_wrapper(command, args):
    try:
        create_save_path(args)
        log = get_logger(args)
        result = command(args)
    except KeyboardInterrupt:
        command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
        if command == 0:
            shutil.rmtree(args.save_dir)
            print("You have successfully delete the created log directory.")
        else:
            print("log directory have been saved.")
    except Exception:
        traceback.print_exc()
        command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
        if command == 0:
            shutil.rmtree(args.save_dir)
            print("You have successfully delete the created log directory.")
        else:
            print("log directory have been saved.")
    return result

def command_decorator(command):
    def wrapper(args):
        try:
            create_save_path(args)
            result = command(args)
        except KeyboardInterrupt:
            command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
            if command == 0:
                shutil.rmtree(args.save_dir)
                print("You have successfully delete the created log directory.")
            else:
                print("log directory have been saved.")
        except Exception:
            traceback.print_exc()
            command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
            if command == 0:
                shutil.rmtree(args.save_dir)
                print("You have successfully delete the created log directory.")
            else:
                print("log directory have been saved.")
        return result
    return wrapper

class MetricDict:
    def __init__(self, metrics) -> None:
        if not isinstance(metrics, dict):
            raise TypeError('MetricDict only container a dict.')
        self.metrics = list(metrics)

    def add_metric(key, value):
        pass

    def _save(self, key):
        pass

    def save_to_tensorboard(self):
        pass