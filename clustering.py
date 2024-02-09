#!/usr/bin/env python3
"""
Name: clutering.py
Description: Clustering by using Word2Vec and Kmeans.
Author: Jinwook Park
Date Created: June 1, 2023
Date Modified: June 10, 2023
Version: 1.0
Python Version: 3.8.5
Dependencies: numpy, nltk, torch, gensim, faiss
License: MIT License
"""
import gensim
import faiss
import pickle


def idx2str(idx, vocab):
    """Index list to string.

    Args:
        idx (list(int)): list of index.
        vocab (str): vocabulary.

    Returns:
        w: list of word.
    """
    w = []
    for i in idx:
        w.append(vocab.idx2word[i])
    return w


def word_embedding(train, valid, test, vocab):
    """Generate Word2Vec Embeddings by using gensim.
    This embeddings are made by whole dataset(train, valid, test).

    Args:
        train (_type_): training dataset.
        valid (_type_): validation dataset.
        test (_type_): test dataset.

    Returns:
        model: Word2Vec model.
    """
    total = train + valid + test
    total = [idx2str(s, vocab) for s in total]

    # Load word2vec model
    model = gensim.models.Word2Vec(
        vector_size=256, window=5, min_count=1, workers=4
    )
    model.build_vocab_from_freq(
        {w: len(vocab) - i for i, w in vocab.idx2word.items()}
    )
    model.train(total, total_examples=len(total), epochs=10)
    return model


def clustering(model):
    """Clustering by using faiss.

    Args:
        model (_type_): Word2Vec model.

    Returns:
        _type_: Kmeans model.
    """
    NUM_CLUSTER = 60
    kmeans = faiss.Kmeans(d=256, k=NUM_CLUSTER, niter=300, verbose=True)
    kmeans.train(model.wv.vectors.astype("float32"))
    return kmeans


if __name__ == "__main__":
    # Load datasets
    with open("edit-eng-std-both.dataset", "rb") as f:
        dataset = pickle.load(f)
    vocab = dataset.word_vocab

    # Load sentences
    train = dataset.train_dataset.get_field("word").content
    valid = dataset.val_dataset.get_field("word").content
    test = dataset.test_dataset.get_field("word").content

    model = word_embedding(train, valid, test, vocab)
    model.save("word2vec-ptb-std.model")
    model.wv.save("word2vec-ptb-std.wordvectors")

    kmeans = clustering(model, train)
    with open("word2vec-ptb-std.means", "wb") as f:
        pickle.dump(kmeans.centroids, f)

    print("finished")
