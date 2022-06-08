from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pickle
import codecs
import json
from scipy import sparse
from scipy.io import savemat, loadmat
import re
import string
import os
import argparse

parser = argparse.ArgumentParser(description='Preparing CNTM Data')
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus: 20ng, webs, tmn, grolier, nyt, reuters, imdb')
parser.add_argument('--data_dir', type=str, default='~/data/dataset/ntm', help='the base dir of datasets')
parser.add_argument('--with_label', type=int, default=0, help='with-1 or without-0 labels (target)')
parser.add_argument('--max_df', type=float, default=0.7, help='maximum document frequency')
parser.add_argument('--min_df', type=int, default=10, help='minimum document frequency')
parser.add_argument('--output_dir', type=str, default='../data', help='the output dir of preprocessed data')
parser.add_argument('--to_text', type=int, default=0, help='1-write to text')

args = parser.parse_args()

class Preprocessor:
    def __init__(self, args):
        self.ds = args.dataset
        self.data_dir = args.data_dir
        self.with_label = args.with_label
        self.to_text = args.to_text
        self.max_df = args.max_df
        self.min_df = args.min_df
        self.path_save = args.output_dir + '/' + args.dataset + '/'
        os.system('mkdir -p ' + self.path_save)
        print('Preparing CNTM Data')

        # Read stopwords
        with open('stops.txt', 'r') as f:
            self.stops = f.read().split('\n')

        self.vocab_size = None
        self.init_docs_tr = None
        self.init_docs_ts = None
        self.init_docs = None
        self.train_target = None
        self.test_target = None
        self.init_targets = None
        self.trSize = 0
        self.vaSize = 100
        self.tsSize = 0
        self.idx_permute = None

        # Read data
        self.read_data()
        # Build vocab
        self.get_vocab()

    def read_data(self):
        print('************* Reading {} data'.format(self.ds))
        if self.ds == 'webs' or self.ds == 'tmn':
            data = loadmat(os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/data.mat'))
            train_data = data['wordsTrain'].transpose().toarray().astype(int)
            test_data = data['wordsTest'].transpose().toarray().astype(int)
            voc = data['vocabulary']
            voc = [v[0][0] for v in voc]

            self.train_target = data['labelsTrain'].squeeze().tolist()
            self.test_target = data['labelsTest'].squeeze().tolist()
            self.init_docs_tr = [[voc[w_id] for w_id, count in enumerate(doc) if count > 0] for doc in train_data]
            self.init_docs_ts = [[voc[w_id] for w_id, count in enumerate(doc) if count > 0] for doc in test_data]
        elif self.ds == 'imdb':
            self.init_docs = []
            base_dir_train_pos = self.data_dir + '/' + self.ds + '_clean/train/pos/'
            base_dir_train_neg = self.data_dir + '/' + self.ds + '_clean/train/neg/'
            base_dir_test_pos = self.data_dir + '/' + self.ds + '_clean/test/pos/'
            base_dir_test_neg = self.data_dir + '/' + self.ds + '_clean/test/neg/'
            train_files_pos = [os.path.expanduser(base_dir_train_pos + file) for file in
                               os.listdir(os.path.expanduser(base_dir_train_pos)) if file.endswith('.txt')]
            train_files_neg = [os.path.expanduser(base_dir_train_neg + file) for file in
                               os.listdir(os.path.expanduser(base_dir_train_neg)) if file.endswith('.txt')]
            test_files_pos = [os.path.expanduser(base_dir_test_pos + file) for file in
                               os.listdir(os.path.expanduser(base_dir_test_pos)) if file.endswith('.txt')]
            test_files_neg = [os.path.expanduser(base_dir_test_neg + file) for file in
                               os.listdir(os.path.expanduser(base_dir_test_neg)) if file.endswith('.txt')]
            for file in train_files_pos:
                with open(file) as f:
                    for line in f:
                        words = re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', line.lower())
                        self.init_docs.append(' '.join(words))
                    f.close()
            for file in train_files_neg:
                with open(file) as f:
                    for line in f:
                        words = re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', line.lower())
                        self.init_docs.append(' '.join(words))
                    f.close()
            for file in test_files_pos:
                with open(file) as f:
                    for line in f:
                        words = re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', line.lower())
                        self.init_docs.append(' '.join(words))
                    f.close()
            for file in test_files_neg:
                with open(file) as f:
                    for line in f:
                        words = re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', line.lower())
                        self.init_docs.append(' '.join(words))
                    f.close()
            pass
        elif self.ds == 'reuters':
            data_file = os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/full.str')
            self.init_docs = []
            self.init_targets = []
            raw_targets = []
            with open(data_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    json_dict = json.loads(line)
                    words = []
                    if 'topics' in json_dict:
                        raw_targets.append(json_dict['topics'][0])
                    else:
                        # raw_targets.append('unknown')
                        continue

                    if 'title' not in json_dict:
                        continue
                    title = re.sub(r'[^a-zA-Z\s]', '', json_dict['title'].lower())
                    title_words = [w for w in re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', title) if w.strip() != '' and w.strip() != ',' and w not in self.stops]
                    words.extend(title_words)
                    if 'body' in json_dict:
                        body = json_dict['body'].lower()
                        body_words = [w for w in re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', body) if w.strip() != '' and w.strip() != ',' and w not in self.stops ]
                        words.extend(body_words)
                    self.init_docs.append(' '.join(words))
            f.close()
            target_set_list = list(set(raw_targets))
            self.init_targets = [target_set_list.index(raw_target) for raw_target in raw_targets]
        elif self.ds == 'nyt':
            data_file = os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/1987.csv')
            self.init_docs = []
            with open(data_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    words = re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', line.lower())
                    self.init_docs.append(' '.join(words))
            f.close()
        elif self.ds == 'grolier':
            data = loadmat(os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/grolier15276.mat'))
            vocab = [v[0] for v in data['words'].squeeze() if v[0] not in self.stops]
            self.vocab_size = len(vocab)
            self.word2id = dict([(w, i) for i, w in enumerate(vocab)])
            self.id2word = dict([(i, w) for i, w in enumerate(vocab)])
            with open(os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/grolier15276.csv'), 'r') as f:
                docs = f.readlines()
            self.init_docs = [' '.join([self.id2word[int(w)] for w in doc.split(',') if w.strip() != '' and int(w) in self.id2word]) for doc in docs]
            f.close()
        else:  # 20ng
            train_data = fetch_20newsgroups(subset='train')
            test_data = fetch_20newsgroups(subset='test')
            self.train_target = train_data.target
            self.test_target = test_data.target
            self.init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_data.data[doc]) for doc in range(len(train_data.data))]
            self.init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_data.data[doc]) for doc in range(len(test_data.data))]

        if self.ds == 'grolier' or self.ds == 'nyt' or self.ds == 'reuters' or self.ds == 'imdb':
            self.trSize = int(np.floor(0.85 * len(self.init_docs)))
            self.tsSize = int(np.floor(0.10 * len(self.init_docs)))
            self.vaSize = int(len(self.init_docs) - self.trSize - self.tsSize)
            self.idx_permute = np.random.permutation(self.trSize + self.vaSize).astype(int)
        else:
            self.trSize = len(self.init_docs_tr) - self.vaSize
            self.tsSize = len(self.init_docs_ts)
            self.idx_permute = np.random.permutation(len(self.init_docs_tr)).astype(int)

            init_docs = self.init_docs_tr + self.init_docs_ts
            init_docs = [[w.lower() for w in init_docs[doc] if not self.contains_punctuation(w)] for doc in range(len(init_docs))]
            init_docs = [[w for w in init_docs[doc] if not self.contains_numeric(w)] for doc in range(len(init_docs))]
            init_docs = [[w for w in init_docs[doc] if len(w) > 1] for doc in range(len(init_docs))]
            self.init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]
            self.init_targets = list(self.train_target) + list(self.test_target)

    def get_vocab(self):
        print('************* Generate Vocab')
        if self.ds != 'grolier':
            print('  counting document frequency of words...')
            cvectorizer = CountVectorizer(min_df=self.min_df, max_df=self.max_df, stop_words=None)
            cvz = cvectorizer.fit_transform(self.init_docs).sign()

            # Get vocabulary
            print('  building the vocabulary...')
            sum_counts = cvz.sum(axis=0)
            v_size = sum_counts.shape[1]
            sum_counts_np = np.zeros(v_size, dtype=int)
            for v in range(v_size):
                sum_counts_np[v] = sum_counts[0, v]
            self.word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
            self.id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
            del cvectorizer
            print('  initial vocabulary size: {}'.format(v_size))

            # Sort elements in vocabulary
            idx_sort = np.argsort(sum_counts_np)
            vocab_aux = [self.id2word[idx_sort[cc]] for cc in range(v_size)]

            # Filter out stopwords (if any)
            vocab_aux = [w for w in vocab_aux if w not in self.stops]
            print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

            # Create dictionary and inverse dictionary
            vocab = vocab_aux
            del vocab_aux
            self.word2id = dict([(w, j) for j, w in enumerate(vocab)])
            self.id2word = dict([(j, w) for j, w in enumerate(vocab)])

        # Remove words not in train_data
        vocab = list(set([w for idx_d in range(self.trSize) for w in self.init_docs[self.idx_permute[idx_d]].split() if w in self.word2id]))
        self.vocab_size = len(vocab)
        self.word2id = dict([(w, j) for j, w in enumerate(vocab)])
        self.id2word = dict([(j, w) for j, w in enumerate(vocab)])

        with open(self.path_save + 'vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        del vocab
        print('  vocabulary after removing words not in train: {}'.format(self.vocab_size))

    def contains_punctuation(self, w):
        return any(char in string.punctuation for char in w)

    def contains_numeric(self, w):
        return any(char.isdigit() for char in w)

    def create_bow_and_save(self, docs, targets, split_name='tr'):
        print('************* Create BOW representations and save disk')
        n_docs = len(docs)  # number of documents in each set

        # create doc indices
        aux = [[j for i in range(len(doc))] for j, doc in enumerate(docs)]
        doc_indices = [int(x) for y in aux for x in y]
        print('  len(np.unique(doc_indices_' + split_name + ')): {} [this should be {}]'.format(
            len(np.unique(doc_indices)), n_docs))
        # create list words
        words = [x for y in docs for x in y]
        print('  len(words_' + split_name + '): ', len(words))
        # create bow representation
        bow = sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, self.vocab_size)).tocsr()

        # split bow to token/value
        indices = [[w for w in bow[doc, :].indices] for doc in range(n_docs)]
        counts = [[c for c in bow[doc, :].data] for doc in range(n_docs)]

        # save to mat file
        savemat(self.path_save + 'bow_' + split_name + '_tokens.mat', {'tokens': indices}, do_compression=True)
        savemat(self.path_save + 'bow_' + split_name + '_counts.mat', {'counts': counts}, do_compression=True)
        if self.with_label == 1:
            savemat(self.path_save + 'bow_' + split_name + '_targets.mat', {'targets': targets}, do_compression=True)
        del docs
        del doc_indices
        del words
        del bow

    def remove_empty(self, in_docs, in_targets, is_test=False):
        out_docs, out_targets = [], []
        for i, doc in enumerate(in_docs):
            if doc == []:
                continue
            # Remove test documents with length=1
            if is_test is True and len(doc) <= 1:
                continue
            out_docs.append(doc)

            if self.with_label == 0 or in_targets is None:
                continue
            out_targets.append(in_targets[i])
        return out_docs, out_targets

    def build(self):
        if self.to_text == 1:
            print('************* save to text')
            if self.ds == 'webs' or self.ds == 'tmn':
                preprocessor.write_tmn_webs_to_text('train') # save to text
                preprocessor.write_tmn_webs_to_text('test') # save to text
            elif self.ds == 'routers':
                preprocessor.write_routers_to_text()  # save to text
            return None

        print('************* Split data')

        # split in train/test/valid
        docs_tr = [[self.word2id[w] for w in self.init_docs[self.idx_permute[idx_d]].split() if w in self.word2id] for idx_d in range(self.trSize)]
        docs_va = [[self.word2id[w] for w in self.init_docs[self.idx_permute[idx_d + self.trSize]].split() if w in self.word2id] for idx_d in range(self.vaSize)]
        docs_ts = [[self.word2id[w] for w in self.init_docs[idx_d + self.trSize + self.vaSize].split() if w in self.word2id] for idx_d in range(self.tsSize)]

        print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), self.trSize))
        print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), self.tsSize))
        print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), self.vaSize))

        # split targets
        target_tr, target_va, target_ts = None, None, None
        if self.with_label == 1:
            target_tr = [self.init_targets[self.idx_permute[idx_d]] for idx_d in range(self.trSize)]
            target_va = [self.init_targets[self.idx_permute[idx_d+self.trSize]] for idx_d in range(self.vaSize)]
            target_ts = [self.init_targets[idx_d+self.trSize+self.vaSize] for idx_d in range(self.tsSize)]
            print('  number of targets (train): {} [this should be equal to {}]'.format(len(target_tr), self.trSize))
            print('  number of targets (test): {} [this should be equal to {}]'.format(len(target_ts), self.tsSize))
            print('  number of targets (valid): {} [this should be equal to {}]'.format(len(target_va), self.vaSize))

        # removing empty documents
        print('  removing empty documents...')
        docs_tr, target_tr = self.remove_empty(docs_tr, target_tr)
        docs_ts, target_ts = self.remove_empty(docs_ts, target_ts, True)
        docs_va, target_va = self.remove_empty(docs_va, target_va)
        print('  splitting test documents in 2 halves...')
        docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
        docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

        self.create_bow_and_save(docs_tr, target_tr, 'tr')
        self.create_bow_and_save(docs_ts, target_ts, 'ts')
        self.create_bow_and_save(docs_ts_h1, target_ts, 'ts_h1')
        self.create_bow_and_save(docs_ts_h2, target_ts, 'ts_h2')
        self.create_bow_and_save(docs_va, target_va, 'va')

        print('Data ready !!')
        print('*************')

    def write_tmn_webs_to_text(self, type='train'):
        data = loadmat(os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/data.mat'))

        voc = data['vocabulary']
        voc = np.array([v[0][0] for v in voc])

        if type == 'test':
            test_data = data['wordsTest'].transpose().toarray()
            target = data['labelsTest'].squeeze().tolist()
            docs = [voc[np.argwhere(doc > 0).squeeze().tolist()] for doc in test_data]
        else: # train
            train_data = data['wordsTrain'].transpose().toarray()
            target = data['labelsTrain'].squeeze().tolist()
            docs = [voc[np.argwhere(doc > 0).squeeze().tolist()] for doc in train_data]

        out_path = os.path.join(self.path_save, '{}_{}_lines.txt'.format(self.ds, type))
        fout = codecs.open(out_path, "w", 'utf-8')
        # write
        for di, doc in enumerate(docs):
            line = " ".join(str(e) for e in doc)
            jsonStr = json.dumps({'text': line, 'label': str(target[di])})
            fout.write('{}\n'.format(jsonStr))
        fout.close()

    def write_routers_to_text(self):
        data_file = os.path.expanduser(self.data_dir + '/' + self.ds + '_clean/full.str')
        fout_tr = codecs.open(os.path.join(self.path_save, '{}_train_lines.txt'.format(self.ds)), "w", 'utf-8')
        fout_ts = codecs.open(os.path.join(self.path_save, '{}_test_lines.txt'.format(self.ds)), "w", 'utf-8')
        with open(data_file, 'r') as f:
            lines = f.readlines()
            total_size = len(lines)
            train_size = 0.85 * total_size
            for i, line in enumerate(lines):
                json_dict = json.loads(line)
                words = []

                if 'topics' in json_dict:
                    target = json_dict['topics'][0]
                else:
                    continue

                if 'title' not in json_dict:
                    continue
                title = re.sub(r'[^a-zA-Z\s]', '', json_dict['title'].lower())
                title_words = [w for w in
                               re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', title) if
                               w.strip() != '' and w.strip() != ',' and w not in self.stops]
                words.extend(title_words)
                if 'body' in json_dict:
                    body = re.sub(r'[^a-zA-Z\s]', '', json_dict['body'].lower())
                    body_words = [w for w in
                                  re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', body) if
                                  w.strip() != '' and w.strip() != ',' and w not in self.stops]
                    words.extend(body_words)

                jsonStr = json.dumps({'text': ' '.join(words), 'label': str(target)})

                if i < train_size:
                    fout_tr.write('{}\n'.format(jsonStr))
                else:
                    fout_ts.write('{}\n'.format(jsonStr))

        f.close()
        fout_tr.close()
        fout_ts.close()

if __name__ == '__main__':
    preprocessor = Preprocessor(args)
    preprocessor.build() # build mat files
    exit(0)
