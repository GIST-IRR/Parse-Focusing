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
    handler = logging.FileHandler(
        os.path.join(
            args.save_dir if path is None else path, f'{log_name}.log'
        ), 'w'
    )
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

def command_decorator(command):
    """If you want to use command decorator as a decorator, you just add decorator to the function that you want to use.
    But, if you want to  use command decorator as a function, you first call the function and then call the function that you want to use with the argument of the function that you called.

    Args:
        command (_type_): _description_
    """
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