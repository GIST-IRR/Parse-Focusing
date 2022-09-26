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
        seed: int,
        python_seed: bool=True,
        numpy_seed: bool=True,
        torch_seed: bool=True,
        cuda_seed: bool=True,
        worker_init_fn: bool=False,
        generator: bool=False,
        deterministic: bool=True,
        benchmark: bool=False,
        work_space_config: str=":16:8"
    ):
    """If you want to fix seed, you have to call this function before assigning
    variables that you want to fix.

    Args:
        seed (int): A random seed that used to fix all seeds.
        python_seed (bool, optional): Fix python random. Defaults to True.
        numpy_seed (bool, optional): Fix numpy random. Defaults to True.
        torch_seed (bool, optional): Fix pytorch random. Defaults to True.
        cuda_seed (bool, optional): Fix CUDA random. Defaults to True.
        worker_init_fn (bool, optional): Return worker_init_fn if True. Defaults to False.
        generator (bool, optional): Return generator if True. Defaults to False.
        deterministic (bool, optional): Set deterministic. Defaults to True.
        benchmark (bool, optional): Set benchmark. Defaults to False.
        work_space_config (_type_, optional): Set work space config. Defaults to ":16:8".

    Returns:
        _type_: _description_
    """
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
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    # CUDA
    if cuda_seed:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.version.cuda >= str(10.2):
            os.environ['CUBLAS_WORKSPACE_CONFIG']=work_space_config
            # or ":4096:2"
        else:
            os.environ['CUDA_LAUNCH_BLOCKING']="1"

    worker_init_fn = None
    generator = None

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
    return saved_name

def get_logger(args, log_name='train',path=None):
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create file handler
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create console handler
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

def reproducible_save(
        model, optimizer, save_name,
        scheduler=None,
        reproducible=True,
        **kwargs
    ):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    checkpoint.update({k: v for k, v in kwargs.items() if v is not None})
    
    if scheduler is not None:
        checkpoint.update({'scheduler': scheduler.state_dict()})
    if reproducible:
        checkpoint.update({
            'random.python': random.getstate(),
            'random.numpy': np.random.get_state(),
            'random.torch': torch.random.get_rng_state(),
            'random.cuda': torch.cuda.get_rng_state(),
            'random.cuda_all': torch.cuda.get_rng_state_all(),
        })
    torch.save(checkpoint, save_name)

def reproducible_load(
        model, optimizer, load_name,
        scheduler=None,
        reproducible=True,
    ):
    # Load checkpoint
    checkpoint = torch.load(load_name)

    # Load model
    model.load_state_dict(checkpoint.pop('model'))
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    
    if scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint.pop('scheduler'))
        except:
            print("Warning: scheduler is not loaded.")

    if reproducible:
        try:
            random.setstate(checkpoint['random.python'])
            np.random.set_state(checkpoint['random.numpy'])
            torch.random.set_rng_state(checkpoint['random.torch'])
            torch.cuda.set_rng_state(checkpoint['random.cuda'])
            torch.cuda.set_rng_state_all(checkpoint['random.cuda_all'])
        except:
            print("Warning: reproducible is not loaded.")

    if scheduler is not None:
        return model, optimizer, scheduler, checkpoint
    else:
        return model, optimizer, checkpoint
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