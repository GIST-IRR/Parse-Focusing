# -*- coding: utf-8 -*-

import argparse
from parser.cmds import Train
from pathlib import Path

from torch_support.train_support import (
    setup_log_dir,
    get_config_from,
    command_decorator,
)
from torch_support.device_support import set_device


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCFGs")
    parser.add_argument("--conf", "-c", default="")
    parser.add_argument("--device", "-d", default="0")
    args = parser.parse_args()

    train(args)
