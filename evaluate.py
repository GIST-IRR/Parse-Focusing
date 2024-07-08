# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

from parser.cmds import Evaluate

from torch_support.train_support import get_config_from
from torch_support.device_support import set_device
from torch_support.reproducibility import fix_seed


def evaluate(args2, device=None):
    if device is not None:
        args2.device = device

    args2.conf = Path(args2.load_from_dir) / "config.yaml"
    args = get_config_from(args2.conf)
    args.update(args2.__dict__)

    # Set the random seed for reproducible experiments
    if hasattr(args, "seed"):
        fix_seed(args.seed, worker_init_fn=False, generator=False)

    args.device = set_device(args.device)

    command = Evaluate()
    command(
        args,
        data_split=args.data_split,
        decode_type=args.decode_type,
        eval_dep=args.eval_dep,
        tag=args.tag,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--eval_dep",
        default=False,
        help="evaluate dependency, only for N(B)L-PCFG",
    )
    parser.add_argument("--data_split", default="test")
    parser.add_argument("--decode_type", default="mbr", help="viterbi or mbr")
    parser.add_argument("--load_from_dir", default="")
    parser.add_argument("--device", "-d", default="0")
    parser.add_argument("--tag", "-t", default="best")
    args = parser.parse_args()
    evaluate(args)
