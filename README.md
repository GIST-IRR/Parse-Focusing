# Parse-focused Neural PCFG

Source code of ACL2024 Findings [Structural Optimization Ambiguity and Simplicity Bias in Unsupervised Neural Grammar Induction](TBA).
Our code edited based on [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG).

## Setup

### prepare environment 

```
conda create -n ungi python=3.11
conda activate ungi
pip install -r requirements.txt
```

### prepare dataset

You can download the dataset and pretrained model (TN-PCFG and NBL-PCFG) from:  https://mega.nz/folder/OU5yiTjC#oeMYj1gBhqm2lRAdAvbOvw

PTB:  ptb_cleaned.zip / CTB and SPRML: ctb_sprml_clean.zip

You can directly use the propocessed pickle file or create pickle file by your own

```
python  preprocessing.py  --train_file path/to/your/file --val_file path/to/your/file --test_file path/to/your/file  --cache path/
```

After this, your data folder should look like this:

```
config/
   ├── tnpcfg_r500_nt250_t500_curriculum0.yaml
   ├── ...
  
data/
   ├── ptb-train-lpcfg.pickle    
   ├── ptb-val-lpcfg.pickle
   ├── ptb-test-lpcfg.pickle
   ├── ...
   
log/
fastNLP/
parser/
train.py
evaluate.py
preprocessing.py
```

## Train

**TN-PCFG**

```
python train.py  --conf tnpcfg_r500_nt250_t500_curriculum0.yaml
```

**Compound PCFG**

```
python train.py --conf cpcfg_nt30_t60_curriculum1.yaml
```

....

## Evaluation

For example, the saved directory should look like this:

```
log/
   ├── NBLPCFG2021-01-26-07_47_29/
   	  ├── config.yaml
   	  ├── best.pt
   	  ├── ...
```

python evaluate.py --load_from_dir log/NBLPCFG2021-01-26-07_47_29  --decode_type mbr --eval_dep 1 

## Out-of-memory

If you encounter OOM, you should adjust the batch size in the yaml file. Normally, for GPUs with 12GB memory, batch size=4~8 is ok, while for evaluation of NBL-PCFGs, you should set a smaller batch size (1 or 2).  

## Pre-processing

`extract_prob.py`: Generate probability distribution based on rule frequency of parse trees in dataset.

```
python -m preprocessing.extract_prob \
--filepath [path_to_dataset] \
--vocab [path_to_vocab] \
--output [path_to_output]
```

`generate_augmented_trees.py`: Add or Replace augmented sentence in dataset. Each augmented sentence generated based on POS tag of tokens in original sentence for given dataset.

```
python -m preprocessing.generate_augmented_trees \
--dataset [path_to_dataset] \
--noise_count [num_to_augmented] \
--noise_threshold [threshold]
```

`generate_focused_parse.py`: Build new dataset that composed with generated parse trees. \[right-branched / left-brancehd / random / right-binarized / left-binarized\] parse trees are generated for each sentence in given dataset.

```
python -m preprocessing.generate_focused_parse.py \
--factor [right-binarized] \
--vocab [vocab/english.vocab] \
--input [path_to_dataset] \
--output [path_to_save]
```

## Post-processing

`string_to_tree.py`: Transform parse trees with string format to NLTK Trees and save to file.

```
python -m postprocessing.string_to_tree \
--filepath "trees/train_seed0.txt" \
--vocab "vocab/english.vocab" \
--output "trees/train_seed0_trees.pt"
```

`tree_to_span.py`: Transform parse trees with string format to spans and save to file.

```
python -m postprocessing.tree_to_span \
--filepath "trees/train_seed0_trees.pt" \
--vocab "vocab/english.vocab" \
--output "trees/train_seed0_span.pt"
```

## Analysis

### Correlation

Each CSV Files have to have the following format:
```
f1 score, likelihood
f1 score, likelihood
...
```

`scatter_with_hist.py`: `Fig. 2(a)` Visualization for correlation between F1 and LL for single model with histogram.

`scatter_comparison.py`: `Fig. 2(b)` Visualization for correlation between F1 and LL for various models.

### Trees

`compare_trees.py`: Calculate F1 score and IoU score for given parse trees.

`rule_frequency.py`: `Fig. 5` Visualize sorted distribution for frequencies that observed rules in parse trees.

`common_uncommon_hist.py`: `Fig. 9` Visualize the degree of rareness for rules and the accuracy according to the degree of rareness.

### The number of Unique rules

`unique_rules_single.py`: `Fig. 3(a)`

`unique_rules_comp.py`: `Fig. 3(b)`

`unique_rules_lang.py`: `Fig. 7`

### Performance

`homo_hetero.py`: `Fig. 6` Visualize the performance according to the combination of multi-parsers.

## Drawing

`ambiguity.py`: `Fig.1` landscape example