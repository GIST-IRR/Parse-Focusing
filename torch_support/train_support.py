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
import json
from easydict import EasyDict

def get_config_from(path, easydict=True, verbose=False):
    # check the path
    if not isinstance(path, Path):
        path = Path(path)
    
    # load the config file
    if path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    elif path.suffix == '.yaml':
        with open(path, 'r') as f:
            config = load(f, Loader=Loader)
    else:  
        raise ValueError(f'Unsupport file type {path.suffix}')
    
    try:
        config.update({'conf': str(path)})
    except:
        raise ValueError(
            f'config file should be a dict, but got {type(config)}'
        )
    # convert to easydict
    if easydict:
        config = EasyDict(config)

    # print the config
    if verbose:
        print(config)
    
    return config

def setup_log_dir(path, parents=True, exist_ok=True):
    if isinstance(path, (list, tuple)):
        path = [Path(p) for p in path]
    else:
        path = [Path(path)]

    for p in path:
        if not p.exists():
            p.mkdir(parents=parents, exist_ok=exist_ok)
            print(f'log dir [{p}] is created.')

def create_save_path(args, copy_config=True, copy_code=True):
    # name setup
    model_name = args.model.model_name
    suffix = \
        f'/{model_name}' \
        + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    # Check existence of the save dir
    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)

    # save the config file and model file.
    if copy_config:
        shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    if copy_code:
        os.makedirs(args.save_dir + "/parser")
        copy_tree("parser/", args.save_dir + "/parser")
    return saved_name

def get_logger(
        args,
        log_name='train', 
        default_level='INFO',
        file_level='INFO',
        console_level='INFO',
        path=None
    ):
    # str to logging level
    default_level = default_level.upper()
    file_level = file_level.upper()
    console_level = console_level.upper()
    
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(default_level)

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
    handler.setLevel(file_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logger.propagate = False
    logger.info(args)
    return logger

def command_decorator(command):
    """If you want to use command decorator as a decorator, you just add decorator to the function that you want to use.
    But, if you want to  use command decorator as a function, you first call the function and then call the function that you want to use with the argument of the function that you called.

    Args:
        command (class or function): main function that you want to use. If you want to use class, you should add __call__ function to the class.
    """
    def wrapper(args):
        try:
            create_save_path(args)
            result = command(args)
            return result
        except KeyboardInterrupt:
            while True:
                cmd = int(input('Do you want to save the model? [y/n]: '))
                if cmd == 'y':
                    print("Log directory have been saved.")
                    break
                elif cmd == 'n':
                    shutil.rmtree(args.save_dir)
                    print("You have successfully delete the created log directory.")
                    break
                else:
                    print("Please enter 'y' or 'n'.")
                    continue
        except Exception:
            traceback.print_exc()
            while True:
                cmd = int(input('Do you want to save the model? [y/n]: '))
                if cmd == 'y':
                    print("Log directory have been saved.")
                    break
                elif cmd == 'n':
                    shutil.rmtree(args.save_dir)
                    print("You have successfully delete the created log directory.")
                    break
                else:
                    print("Please enter 'y' or 'n'.")
                    continue
    return wrapper