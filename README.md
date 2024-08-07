# Parse-focused Neural PCFG

Source code of ACL2024 Findings [Structural Optimization Ambiguity and Simplicity Bias in Unsupervised Neural Grammar Induction](https://arxiv.org/abs/2407.16181).
Our code edited based on [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG).

## Overview

![Overview](images/overview_pfnpcfg_bg.jpg)

## Setup

### Prepare environment 

```bash
conda create -n swpf python=3.11
conda activate swpf
pip install -r requirements.txt
```

### Prepare dataset

If you need to download the datasets, please refer to [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG).

You can directly use the propocessed pickle file or create pickle file by your own

```bash
python -m preprocessing.preprocessing \
--dir [path_to_dir_that_contain_dataset_txt]
--save_dir [path_to_save_dataset]
```

### (Optional) Prepare pre-trained model

You can download our pre-trained model from [here](TBA) (TBA).

### (Optional) Generating dataset for baseline

Build new dataset that composed with generated parse trees. \[`right-branched` / `left-brancehd` / `random` / `right-binarized` / `left-binarized`\] parse trees are generated for each sentence in given dataset.

```bash
python -m preprocessing.generate_focused_parse \
--factor [right-binarized/left-binarized/random/right-branched/left-branched] \
--vocab [path_to_vocab] \
--input [path_to_dataset] \
--output [path_to_save]
```

If you want to generate datasets for all languages, factors, and splits (train, valid, test):

```bash
./generate_focused_parse.sh
```

You can include or exclude options for languages, factors and splits in script.

## Train

**Parse-focused TN-PCFG**

```bash
python train.py \
--conf pftnpcfg_r500_nt250_t500_curriculum0.yaml
```

## Evaluation

```bash
python evaluate.py \
--load_from_dir "[PATH TO LOG]" \
--decode_type mbr \
--eval_dep 1 
```

## Paring

```bash
python parse.py --load_from_dir
```

## Out-of-memory

If you encounter OOM, you should adjust the batch size in the yaml file. Normally, for GPUs with 12GB memory, batch size=4~8 is ok, while for evaluation of NBL-PCFGs, you should set a smaller batch size (1 or 2).  

## Post-processing

### String to Tree

Transform parse trees with string format to NLTK Trees and save to file.

```bash
python -m postprocessing.string_to_tree \
--filepath "trees/train_seed0.txt" \
--vocab "vocab/english.vocab" \
--output "trees/train_seed0_trees.pt"
```

### Tree to Span

Transform parse trees with string format to spans and save to file.

```bash
python -m postprocessing.tree_to_span \
--filepath "trees/train_seed0_trees.pt" \
--vocab "vocab/english.vocab" \
--output "trees/train_seed0_span.pt"
```

## Analysis

### Correlation between F1 and NLL

Each CSV Files have to have the following format:
```
f1 score, likelihood
f1 score, likelihood
...
```

`scatter_with_hist.py`: `Fig. 2(a)` Visualization for correlation between F1 and LL for single model with histogram.

`scatter_comparison.py`: `Fig. 2(b)` Visualization for correlation between F1 and LL for various models.

### Trees

(최종 정리 필요) `compare_trees.py`: `Tab. 1` Calculate F1 score and IoU score for given parse trees.

`rule_frequency.py`: `Fig. 5` Visualize sorted distribution for frequencies that observed rules in parse trees.

(정리 필요) `common_uncommon_hist.py`: `Fig. 9` Visualize the degree of rareness for rules and the accuracy according to the degree of rareness.

### The number of Unique rules

Visualize the number of unique rules for each sentence length.

#### For single model in figure. (`Fig. 3(a)`) 

```bash
python3 -m analyzer.unique_rules \
--input "[CSV file path]" \
--output "[Target output file]"
```

#### For different models in same figure. (`Fig. 3(b)`)

Use same command with `Fig. 3(a)`, but CSV file have to involve `group id` column to distinguish each group.

#### For each language in different sub-figures. (`Fig. 7`)

The column `group id` represent as subtitle of figure.
The following `tick_size`, `legend_size`, `label_size` is recommended for this figure.

```bash
python3 -m analyzer.unique_rules \
--input "[CSV file path]" \
--output "[Target output file path]" \
--split_by_group \
--n_col 5 \
--n_row 2 \
--tick_size 17 \
--legend_size 17 \
--label_size 30
```

### Performance

Visualize the performance according to the combination of multi-parsers.

#### For absolute performance (`Fig. 10`)

```bash
python3 -m analyzer.homo_hetero \
--input "[CSV file path]" \
--output "[output file path]" \
```

#### For difference between pre-trained parsers and trained models (`Fig. 6`)

```bash
python3 -m analyzer.homo_hetero \
--input "[CSV file path]" \
--output "[output file path]" \
--difference
```