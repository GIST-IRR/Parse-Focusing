import argparse
import os
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from easydict import EasyDict as edict

import torch

from parser.cmds import Evaluate


def multi_evaluate(args):
    files = sorted(os.listdir(args.load_from_dir))
    log_file = os.path.join(args.load_from_dir, f"{args.decode_type}-test.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    for f in files:
        path = os.path.join(args.load_from_dir, f)
        if os.path.isdir(path):
            yaml_cfg = yaml.load(
                open(path + "/config.yaml", "r"), Loader=Loader
            )
            cfg = edict(yaml_cfg)
            cfg.load_from_dir = path
            print(f"Set the device with ID {args.device} visible")
            # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            cfg.device = (
                f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
            )

            command = Evaluate()
            command(
                cfg,
                decode_type=args.decode_type,
                eval_dep=args.eval_dep,
                data_split=args.data_split,
                tag=args.tag,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dep", default=False)
    parser.add_argument("--load_from_dir", "-d", required=True, default="log")
    parser.add_argument("--decode_type", default="mbr")
    parser.add_argument("--device", default="0")
    parser.add_argument("--tag", default="best")
    parser.add_argument("--data_split", default="test")
    args = parser.parse_args()

    multi_evaluate(args)
