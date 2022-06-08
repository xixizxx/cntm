# coding=utf-8
import os
import random
import json5
import torch
from torch import optim
import numpy as np
from datetime import datetime
from .utils.loader import get_vocab, get_batch, fetch_data
from .utils.plot import nearest_neighbors
from .model import CNTM
from .evaluator import Evaluator


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        self.vocab = get_vocab(os.path.join(self.args.data_path))
        self.args.vocab_size = len(self.vocab)
        self.embeddings = self.get_embeddings()

        train = fetch_data(os.path.join(self.args.data_path), 'train', self.args.cl)
        self.train_tokens = train['tokens']
        self.train_counts = train['counts']
        self.args.num_docs_train = len(self.train_tokens)
        self.train_targets = None
        if 'targets' in train:
            self.train_targets = train['targets']

        self.model, self.optimizer, self.ckpt = self.build_model()

    def run(self):
        print('=*' * 100)
        print('Training a Contrastive Neural Topic Model on {} with the following settings: {}'.format(self.args.dataset.upper(), self.args))
        print('=*' * 100)
        ## define checkpoint
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        print('\n')
        print('Visualizing model quality before training...')
        # visualize(model)
        print('\n')

        evaluator = Evaluator(self.args)

        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch)
            val_ppl = evaluator.evaluate(self.model, 'val')
            if val_ppl < best_val_ppl:
                with open(self.ckpt, 'wb') as f:
                    torch.save(self.model, f)
                best_epoch = epoch
                best_val_ppl = val_ppl
            else:
                ## check whether to anneal lr
                lr = self.optimizer.param_groups[0]['lr']
                if self.args.anneal_lr and (
                        len(all_val_ppls) > self.args.nonmono and val_ppl > min(all_val_ppls[:-self.args.nonmono]) and lr > 1e-5):
                    self.optimizer.param_groups[0]['lr'] /= self.args.lr_factor
            # if epoch % args.visualize_every == 0:
            #     visualize(model)
            all_val_ppls.append(val_ppl)
        with open(self.ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(self.device)
        val_ppl = evaluator.evaluate(model, 'val')

    def train(self, epoch):
        start_time = datetime.now()
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        acc_cl_loss = 0
        cnt = 0
        indices = torch.randperm(self.args.num_docs_train)
        indices = torch.split(indices, self.args.batch_size)
        for idx, ind in enumerate(indices):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            data_batch, target_batch = get_batch(self.train_tokens, self.train_counts, ind, self.args.vocab_size, self.device,
                                                 self.train_targets)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta, cl_loss = self.model(data_batch, normalized_data_batch, target_batch, self.args.cl, self.args.label)
            total_loss = recon_loss + kld_theta + cl_loss
            total_loss.backward()

            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            acc_cl_loss += torch.sum(cl_loss).item()
            cnt += 1

            if idx % self.args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_cl_loss = round(acc_cl_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta + cur_cl_loss, 2)

                print(
                    'Epoch: {} .. batch: {}/{} .. LR: {} .. CL_loss: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                        epoch, idx, len(indices), self.optimizer.param_groups[0]['lr'], cur_cl_loss, cur_kl_theta, cur_loss,
                        cur_real_loss))

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_cl_loss = round(acc_cl_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta + cur_cl_loss, 2)
        end_time = datetime.now()
        interval_time = str(end_time - start_time)
        print('*' * 100)
        print('Epoch----->{} ({}).. LR: {} .. CL_loss: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, interval_time, self.optimizer.param_groups[0]['lr'], cur_cl_loss, cur_kl_theta, cur_loss, cur_real_loss))
        print('*' * 100)

    def visualize(self, m, show_emb=True):
        if not os.path.exists('./results'):
            os.makedirs('./results')

        m.eval()

        queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love',
                   'intelligence', 'money', 'politics', 'health', 'people', 'family']

        ## visualize topics using monte carlo
        with torch.no_grad():
            print('#' * 100)
            print('Visualize topics...')
            topics_words = []
            gammas, _, _ = m.get_beta()
            for k in range(self.args.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.args.num_words + 1:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                print('Topic {}: {}'.format(k, topic_words))

            if show_emb:
                ## visualize word embeddings by using V to get nearest neighbors
                print('#' * 100)
                print('Visualize word embeddings by using output embedding matrix')
                try:
                    embeddings = m.rho.weight  # Vocab_size x E
                except:
                    embeddings = m.rho  # Vocab_size x E
                neighbors = []
                for word in queries:
                    print('word: {} .. neighbors: {}'.format(
                        word, nearest_neighbors(word, embeddings, self.vocab)))
                print('#' * 100)

    def get_embeddings(self):
        embeddings = None
        if not self.args.train_embeddings:
            emb_path = self.args.emb_path
            vect_path = os.path.join(self.args.data_path.split('/')[0], 'embeddings.pkl')
            vectors = {}
            with open(emb_path, 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    if word in self.vocab:
                        vect = np.array(line[1:]).astype(np.float)
                        vectors[word] = vect
            embeddings = np.zeros((self.args.vocab_size, self.args.emb_size))
            words_found = 0
            for i, word in enumerate(self.vocab):
                try:
                    embeddings[i] = vectors[word]
                    words_found += 1
                except KeyError:
                    embeddings[i] = np.random.normal(scale=0.6, size=(self.args.emb_size,))
            embeddings = torch.from_numpy(embeddings).to(self.device)
            self.args.embeddings_dim = embeddings.size()
        return embeddings

    def build_model(self):
        if self.args.cl == 1:
            model = CNTM(self.args.num_topics, self.args.vocab_size, self.args.t_hidden_size, self.args.rho_size, self.args.emb_size,
                         self.args.theta_act, self.embeddings, self.args.train_embeddings, self.args.enc_drop,
                         self.args.cl, self.args.tau).to(self.device)
        else:
            model = CNTM(self.args.num_topics, self.args.vocab_size, self.args.t_hidden_size, self.args.rho_size, self.args.emb_size,
                         self.args.theta_act, self.embeddings, self.args.train_embeddings, self.args.enc_drop).to(self.device)
        print('model: {}'.format(model))

        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'asgd':
            optimizer = optim.ASGD(model.parameters(), lr=self.args.lr, t0=0, lambd=0., weight_decay=self.args.weight_decay)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr)

        ckpt = os.path.join(self.args.save_path, 'cntm_{}_epc_{}_K_{}_CL_{}_TAU_{}_LABEL_{}'.format(
            self.args.dataset, self.args.epochs, self.args.num_topics, self.args.cl, self.args.tau, self.args.label))
        return  model, optimizer, ckpt


class EarlyStop(Exception):
    pass
