import numpy as np
import random

def loadVocab(fname):
    '''
    vocab = {"<PAD>": 0, ...}
    idf   = { 0: log(total_doc/doc_freq)}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for index, word in enumerate(f):
            word = word.strip()
            vocab[word] = index
    return vocab

def toVec(tokens, vocab):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["_UNK_"])
    return length, np.array(vec)


def loadDataset(premise_file, hypothesis_file, label_file, vocab, maxlen):
    # premise
    premise_tokens = []
    premise_vec = []
    premise_len = []
    with open(premise_file, 'rt') as f1:
        for line in f1:
            line = line.strip()
            p_tokens = line.split(' ')[:maxlen]
            p_len, p_vec = toVec(p_tokens, vocab)
            premise_tokens.append(p_tokens)
            premise_vec.append(p_vec)
            premise_len.append(p_len)

    # hypothesis
    hypothesis_tokens = []
    hypothesis_vec = []
    hypothesis_len = []
    with open(hypothesis_file, 'rt') as f2:
        for line in f2:
            line = line.strip()
            h_tokens = line.split(' ')[:maxlen]
            h_len, h_vec = toVec(h_tokens, vocab)
            hypothesis_tokens.append(h_tokens)
            hypothesis_vec.append(h_vec)
            hypothesis_len.append(h_len)

    # label
    label = []
    with open(label_file, 'rt') as f3:
        for line in f3:
            line = line.strip()
            label.append(int(line))

    assert len(premise_tokens) == len(hypothesis_tokens)
    assert len(hypothesis_tokens) == len(label)

    # dataset
    dataset = []
    for i in range(len(label)):
        dataset.append( (premise_tokens[i], premise_vec[i], premise_len[i],
                         label[i],
                         hypothesis_tokens[i], hypothesis_vec[i], hypothesis_len[i]) )
    return dataset

def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec

def batch_iter(data, batch_size, num_epochs, maxlen, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            x_premise = []
            x_hypothesis = []
            x_premise_len = []
            x_hypothesis_len = []
            targets = []

            for rowIdx in range(start_index, end_index):
                premise_tokens, premise_vec, premise_len,\
                label, \
                hypothesis_tokens, hypothesis_vec, hypothesis_len = data[rowIdx]

                # normalize premise_vec and hypothesis_vec
                new_premise_vec = normalize_vec(premise_vec, maxlen)    # pad the original vec to the same maxlen
                new_hypothesis_vec = normalize_vec(hypothesis_vec, maxlen)

                x_premise.append(new_premise_vec)
                x_premise_len.append(premise_len)
                x_hypothesis.append(new_hypothesis_vec)
                x_hypothesis_len.append(hypothesis_len)
                targets.append(label)

            yield np.array(x_premise), np.array(x_hypothesis), np.array(x_premise_len), np.array(x_hypothesis_len),np.array(targets)
