#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/pytorch

python evalb.py \
    --evalb ./EVALB \
    --evalb_config ./EVALB/diora.prm \
    --out ./evalb/mlp-softmax \
    --pred ./evalb/mlp-softmax/parse.jsonl \
    --gold ./data/ptb/english-test.txt