from pathlib import Path
import argparse

from nltk import Tree
import pickle
import torch

from utils import tree_to_span, clean_word


def main(filepath, vocab, output):
    with Path(filepath).open("r") as f:
        trees = f.readlines()

    with Path(vocab).open("rb") as f:
        vocab = pickle.load(f)

    trees = [Tree.fromstring(t) for t in trees]
    new_trees = []
    for tree in trees:
        sentence = tree.leaves()
        word = clean_word(sentence)
        word = [vocab[w] for w in word]
        span = tree_to_span(tree)
        span = [(s[0], s[1]) for s in span]
        word_dict = {"sentence": sentence, "word": word, "tree": span}
        new_trees.append(word_dict)

    with Path(output).open("wb") as f:
        torch.save(new_trees, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath", type=str, default="raw_trees/diora_train_s1.txt"
    )
    parser.add_argument("--vocab", type=str, default="vocab/english.vocab")
    parser.add_argument(
        "--output", type=str, default="trees/tmp_diora_english_s1.pt"
    )
    args = parser.parse_args()
    main(args.filepath, args.vocab, args.output)
