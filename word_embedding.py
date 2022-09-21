import gensim
import pickle

# Load datasets
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
valid = dataset.val_dataset.get_field('word').content
test = dataset.test_dataset.get_field('word').content

total = train + valid + test
total = [idx2str(s, vocab) for s in total]

# Load word2vec model
model = gensim.models.Word2Vec(vector_size=256, window=5, min_count=1, workers=4)
model.build_vocab_from_freq({w: len(vocab)-i for i, w in vocab.idx2word.items()})
model.train(total, total_examples=len(total), epochs=10)
model.save('word2vec-ptb-std.model')
model.wv.save('word2vec-ptb-std.wordvectors')