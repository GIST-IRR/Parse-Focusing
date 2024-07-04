#!/bin/usr/env python3
import argparse
import pickle
from pathlib import Path


def main(vocab, data_path, factor):
    # Vocabulary
    thr = 60
    with Path(vocab).open("rb") as f:
        vocab = pickle.load(f)

    def unknown_masking(vocab, thr):
        unk_count = 0
        for token, t_id in vocab:
            if t_id >= thr:
                unk_count += vocab.word_count[token]
        vocab = vocab[:thr]
        vocab.insert(0, ("<unk>", unk_count))
        return vocab

    vocab = unknown_masking(vocab, thr)
    # vocab_ratio = [(e[0], e[1] / vocab_sum) for e in vocab]

    print(f"Vocabulary size: {len(vocab)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab",
        default="vocab/english.vocab",
        type=str,
    )
    parser.add_argument(
        "--dataset", type=str, default="data/data.clean/edit-english-train.txt"
    )
    parser.add_argument("--factor", type=str, default=None)
    args = parser.parse_args()

    main(args.vocab, args.dataset, args.factor)
