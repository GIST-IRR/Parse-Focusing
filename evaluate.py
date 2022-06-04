# -*- coding: utf-8 -*-

import os
from parser.cmds import Evaluate
import torch
from easydict import EasyDict as edict
import yaml
import click


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

    command = Evaluate()
    command(args, data_split=data_split, decode_type=decode_type, eval_dep=eval_dep)


if __name__ == '__main__':
    main()