import os
import pickle
import numpy as np
import torch 
import scipy.io
from scipy.io import loadmat

def fetch_data(path, name, cl):
    ret = {}

    target_file = None
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
        target_file = os.path.join(path, 'bow_tr_targets.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
        target_file = os.path.join(path, 'bow_va_targets.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
        target_file = os.path.join(path, 'bow_ts_targets.mat')

    ret['tokens'] = scipy.io.loadmat(token_file)['tokens'].squeeze()
    ret['counts'] = scipy.io.loadmat(count_file)['counts'].squeeze()
    if target_file is not None and os.path.exists(target_file):
        ret['targets'] = scipy.io.loadmat(target_file, squeeze_me=True)['targets']

    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        ret['tokens_1'] = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        ret['counts_1'] = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        ret['tokens_2'] = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        ret['counts_2'] = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        target_1_file = os.path.join(path, 'bow_ts_h1_targets.mat')
        target_2_file = os.path.join(path, 'bow_ts_h2_targets.mat')
        if os.path.exists(target_1_file) and os.path.exists(target_2_file):
            ret['targets_1'] = scipy.io.loadmat(target_1_file, squeeze_me=True)['targets']
            ret['targets_2'] = scipy.io.loadmat(target_2_file, squeeze_me=True)['targets']

    return ret

def group_short_texts(docs, labels, group_size=5):
    out_docs = []
    out_labels = []

    short_docs_per_label = {}
    count = {}
    for i, label in enumerate(labels):
        doc = docs[i]

        if label not in short_docs_per_label.keys():
            short_docs_per_label[label] = np.zeros(len(doc))
            count[label] = 0

        if count[label] >= group_size:
            out_docs.append(np.array(short_docs_per_label[label]))
            out_labels.append(label)
            short_docs_per_label[label] = np.zeros(len(doc))
            count[label] = 0

        short_docs_per_label[label] += doc
        count[label] += 1

    return np.array(out_docs), np.array(out_labels)

def _fetch_tmn_webs(data_path, is_to_dense=True):
    data = loadmat(data_path + '/data.mat')

    wordsTrainOrig = data['wordsTrain'].transpose().toarray().astype(int)
    wordsTestOrig = data['wordsTest'].transpose().toarray().astype(int)
    labelsTrainOrig = data['labelsTrain'].squeeze()
    labelsTestOrig = data['labelsTest'].squeeze()

    wordsTrain, labelsTrain = group_short_texts(wordsTrainOrig, labelsTrainOrig)
    wordsTest, labelsTest = group_short_texts(wordsTestOrig, labelsTestOrig)

    idx_permute_tr = np.random.permutation(len(wordsTrain)).astype(int)
    idx_permute_ts = np.random.permutation(len(wordsTest)).astype(int)
    train_data, train_target = [], []
    for idx in idx_permute_tr:
        train_data.append(wordsTrain[idx])
        train_target.append(labelsTrain[idx])
    train_data, train_target = np.array(train_data), np.array(train_target)

    test_data, test_target = [], []
    for idx in idx_permute_ts:
        test_data.append(wordsTest[idx])
        test_target.append(labelsTest[idx])
    test_data, test_target = np.array(test_data), np.array(test_target)

    tokens_tr = [np.array([np.transpose(np.argwhere(doc > 0).squeeze())]) for doc in train_data[:(train_data.shape[0]-100)]]
    tokens_va = [np.array([np.transpose(np.argwhere(doc > 0).squeeze())]) for doc in train_data[(train_data.shape[0]-100):]]
    tokens_ts = [np.array([np.transpose(np.argwhere(doc > 0).squeeze())]) for doc in test_data]
    counts_tr = [np.array([doc[np.argwhere(doc > 0).squeeze().tolist()]]) for doc in train_data[:(train_data.shape[0]-100)]]
    counts_va = [np.array([doc[np.argwhere(doc > 0).squeeze().tolist()]]) for doc in train_data[(train_data.shape[0]-100):]]
    counts_ts = [np.array([doc[np.argwhere(doc > 0).squeeze().tolist()]]) for doc in test_data]
    targets_tr = [target for target in train_target[:(train_target.shape[0]-100)]]
    targets_va = [target for target in train_target[(train_target.shape[0]-100):]]
    targets_ts = [target for target in test_target]

    tokens_ts1, tokens_ts2, counts_ts1, counts_ts2 = [], [], [], []
    for i, tokens in enumerate(tokens_ts):
        tokens = tokens.squeeze()
        if len(tokens.shape) == 0:
            tokens = [tokens]
        counts = counts_ts[i].squeeze()
        if len(counts.shape) == 0:
            counts = [counts]
        for j, w in enumerate(tokens):
            if j <= len(tokens)/2.0-1:
                tokens_ts1.append(w)
                counts_ts1.append(counts[j])
            else:
                tokens_ts2.append(w)
                counts_ts2.append(counts[j])
    # tokens_ts1 = [[w for i,w in enumerate(tokens) if i<=len(tokens)/2.0-1] for tokens in tokens_ts]
    # tokens_ts2 = [[w for i,w in enumerate(tokens) if i>len(tokens)/2.0-1] for tokens in tokens_ts]
    # counts_ts1 = [[w for i,w in enumerate(counts) if i<=len(counts)/2.0-1] for counts in counts_ts]
    # counts_ts2 = [[w for i,w in enumerate(counts) if i>len(counts)/2.0-1] for counts in counts_ts]

    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]

    train = {'tokens': tokens_tr, 'counts': counts_tr, 'targets': targets_tr}
    valid = {'tokens': tokens_va, 'counts': counts_va, 'targets': targets_va}
    test = {'tokens': tokens_ts, 'counts': counts_ts, 'targets': targets_ts,
            'tokens_1': tokens_ts1, 'counts_1': counts_ts1, 'targets_1': targets_ts,
            'tokens_2': tokens_ts2, 'counts_2': counts_ts2, 'targets_2': targets_ts
            }
    return train, valid, test, voc

def get_vocab(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def get_batch(tokens, counts, ind, vocab_size, device, targets):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    target_batch = []
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        if targets is not None:
            target = targets[doc_id]
            target_batch.append(target)
        # if type(doc) == np.ndarray:
        #     if doc.size == 1:
        #         doc = [doc]
        #     else:
        #         doc = doc.squeeze()
        # elif type(doc) == list:
        #     pass
        # else:
        #     doc = [doc]
        #
        # if type(count) == np.ndarray:
        #     if count.size == 1:
        #         count = [count]
        #     else:
        #         count = count.squeeze()
        # elif type(count) == list:
        #     pass
        # else:
        #     count = [count]
        # L = count.shape[1]
        if len(doc.shape) == 0:
            doc = np.array([doc])
        if len(count.shape) == 0:
            count = np.array([count])
        # L = count.shape[1]
        if len(doc) == 1:
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    target_batch = torch.tensor(target_batch).to(device)
    return data_batch, target_batch
