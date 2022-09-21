import numpy as np
import torch
import gensim

import nltk
from nltk.cluster import KMeansClusterer

import faiss

import pickle

with open('edit-eng-std-both.dataset', 'rb') as f:
    dataset = pickle.load(f)
vocab = dataset.word_vocab

def idx2str(idx, vocab):
    w = []
    for i in idx:
        w.append(vocab.idx2word[i])
    return w

# Load sentences
train = dataset.train_dataset.get_field('word').content
# valid = dataset.val_dataset.get_field('word').content
# test = dataset.test_dataset.get_field('word').content

model = gensim.models.Word2Vec.load('word2vec-ptb-std.model')

# def sentence_vectorizer(sent, model):
#     sent_vec = []
#     numw = 0
#     for w in sent:
#         try:
#             if numw == 0:
#                 sent_vec = model.wv[w]
#             else:
#                 sent_vec = np.add(sent_vec, model.wv[w])
#             numw+=1
#         except:
#             pass
#     return np.asarray(sent_vec) / numw

# # Vectorize sentences
# train_vec = [sentence_vectorizer(sent, model) for sent in train]

NUM_CLUSTER = 60
n_init = 60
# kl_clusterer = KMeansClusterer(NUM_CLUSTER, distance=nltk.cluster.util.cosine_distance, repeats=25)
# assigned_clusters = kl_clusterer.cluster(train_vec, assign_clusters=True)
kmeans = faiss.Kmeans(d=256, k=NUM_CLUSTER, niter=300, verbose=True)
kmeans.train(model.wv.vectors.astype('float32'))

with open('word2vec-ptb-std.means', 'wb') as f:
    pickle.dump(kmeans.centroids, f)

print('finished')