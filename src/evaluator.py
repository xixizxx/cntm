# coding=utf-8
import os
import math
import torch
import numpy as np
from .utils.loader import get_vocab, get_batch, fetch_data
from .utils.plot import nearest_neighbors, plot_word_points, plot_embedding, plot_embedding_specific
from .utils.common import calc_tc, calc_topic_coherence, calc_topic_diversity


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        self.ckpt = args.load_from
        self.vocab = get_vocab(os.path.join(args.data_path))
        self.args.vocab_size = len(self.vocab)

        # 1. training data
        train = fetch_data(os.path.join(self.args.data_path), 'train', self.args.cl)
        self.train_tokens = train['tokens']
        self.train_counts = train['counts']
        self.args.num_docs_train = len(self.train_tokens)
        self.train_targets = None
        if 'targets' in train:
            self.train_targets = train['targets']

        # 2. dev set
        valid = fetch_data(os.path.join(self.args.data_path), 'valid', self.args.cl)
        self.valid_tokens = valid['tokens']
        self.valid_counts = valid['counts']
        args.num_docs_valid = len(self.valid_tokens)
        self.valid_targets = None
        if 'targets' in valid:
            self.valid_targets = valid['targets']

        # 3. test data
        test = fetch_data(os.path.join(self.args.data_path), 'test', self.args.cl)
        self.test_tokens = test['tokens']
        self.test_counts = test['counts']
        args.num_docs_test = len(self.test_tokens)
        self.test_1_tokens = test['tokens_1']
        self.test_1_counts = test['counts_1']
        args.num_docs_test_1 = len(self.test_1_tokens)
        self.test_2_tokens = test['tokens_2']
        self.test_2_counts = test['counts_2']
        args.num_docs_test_2 = len(self.test_2_tokens)
        self.test_targets = None
        self.test_1_targets = None
        self.test_2_targets = None
        if 'targets' in test:
            self.test_targets = test['targets']
            self.test_1_targets = test['targets_1']
            self.test_2_targets = test['targets_2']

    def run(self):
        with open(self.ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            ## get document completion perplexities
            test_ppl = self.evaluate(model, 'test', tc=self.args.tc, td=self.args.td)
            print('Test PPL: {}'.format(test_ppl))

            # for plot: the top prob topic per doc
            top_topic_of_doc = []
            doc_word_indices = []

            ## get most used topics
            indices = torch.tensor(range(self.args.num_docs_train))
            indices = torch.split(indices, self.args.batch_size)
            thetaAvg = torch.zeros(1, self.args.num_topics).to(self.device)
            thetaWeightedAvg = torch.zeros(1, self.args.num_topics).to(self.device)
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch, target_batch = get_batch(self.train_tokens, self.train_counts, ind, self.args.vocab_size, self.device, self.train_targets)
                sums = data_batch.sum(1).unsqueeze(1)
                cnt += sums.sum(0).squeeze().cpu().numpy()
                if self.args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                theta, _ = model.get_theta(normalized_data_batch)

                # find each doc's top topic for plot
                doc_word_indices.extend([np.argwhere(doc > 0).squeeze() for doc in data_batch.cpu()])
                for doc_topics in theta:
                    max_topic = max(doc_topics)
                    top_topic_of_doc.append(doc_topics.tolist().index(max_topic.item()))

                thetaAvg += theta.sum(0).unsqueeze(0) / self.args.num_docs_train
                weighed_theta = sums * theta
                thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
                if idx % 100 == 0 and idx > 0:
                    print('batch: {}/{}'.format(idx, len(indices)))
            thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
            ## most used topics
            print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

            ## show topics
            beta, t_emb, w_emb = model.get_beta()
            topic_indices = list(np.random.choice(self.args.num_topics, 10))  # 10 random topics
            print('\nAll topics and their top words\n')
            for k in range(self.args.num_topics):  # topic_indices:
                gamma = beta[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.args.num_words + 1:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
                print('Topic {}: {}'.format(k, topic_words))

            if self.args.train_embeddings:
                ## show etm embeddings
                # try:
                #     rho = model.rho.weight.cpu()
                # except:
                #     rho = model.rho.cpu()
                # queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love',
                #            'intelligence', 'money', 'politics', 'health', 'people', 'family']
                # print('\n')
                # print('Learned embeddings...')
                # for word in queries:
                #     print('word: {} .. neighbors: {}'.format(word, nearest_neighbors(word, rho, self.vocab)))
                print('\n')

            ## Plot
            if self.args.plot_type == 1:
                print('\n1. Plot top 10 word embeddings of each topic\n')
                plot_word_points(beta, self.vocab, w_emb, t_emb, self.args.type_num)
            elif self.args.plot_type == 2:
                print('\n2. Plot all doc embedding points by their topics\n')
                # doc_topic_embeddings = [t_emb[topic_idx].tolist() for topic_idx in top_topic_of_doc]
                doc_embeddings = []
                for word_indices in doc_word_indices:
                    if len(word_indices.shape) == 0:
                        word_indices = torch.stack([word_indices])
                    doc_embeddings.append(torch.mean(torch.stack([w_emb[w_i] for w_i in word_indices]), dim=0).tolist())
                # content_size = len(doc_embeddings)
                # for topic_embedding in t_emb:
                #     doc_embeddings.append(topic_embedding.tolist())
                plot_embedding(doc_embeddings, top_topic_of_doc, self.args.type_num)
            elif self.args.plot_type == 3:
                print('\n3. Plot all word embedding points by their topics\n')
                topic_embeddings = []
                top_topic_of_word = []
                for i, word_topics in enumerate(beta.transpose(0, 1)):
                    max_topic = max(word_topics)
                    top_topic_of_word.append(word_topics.tolist().index(max_topic.item()))
                plot_embedding(w_emb.cpu(), top_topic_of_word, self.args.type_num, 't-SNE vocabulary distribution with topics')
            elif self.args.plot_type == 4:
                print('\n4. Plot queried word embeddings nearby\n')
                # queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love',
                #            'intelligence', 'money', 'politics', 'health', 'people', 'family']
                # queries = ['woman', 'computer', 'sports', 'health', 'war', 'love', 'politics', 'movie']
                queries = ['computer', 'sports', 'health', 'war', 'love',
                           'religion', 'politics', 'family', 'job', 'movie']
                # queries = ['world', 'usa', 'sport', 'business', 'science', 'entertainment', 'health']
                queried_words_embedding = []
                full_neighbors = []
                full_indices = []
                full_targets = []
                for i, word in enumerate(queries):
                    neighbors, neighbor_indices = nearest_neighbors(word, w_emb, self.vocab)
                    print('Word embeddings...')
                    print('word: {} .. neighbors: {}'.format(word, neighbors))
                    full_neighbors.extend(neighbors)
                    full_indices.extend(neighbor_indices)
                    queried_words_embedding.extend([w_emb[w_i].tolist() for w_i in neighbor_indices])
                    full_targets.extend([i] * len(neighbor_indices))
                plot_embedding_specific(self.vocab, queried_words_embedding, None, full_targets, full_neighbors, )
                # plot_embedding_specific(vocab, w_emb.cpu(), full_indices, full_targets,
                #                         't-SNE query words distribution with topics')

    def evaluate(self, m, source, tc=False, td=False):
        """Compute perplexity on document completion.
        """
        m.eval()
        with torch.no_grad():
            if source == 'val':
                indices = torch.split(torch.tensor(range(self.args.num_docs_valid)), self.args.eval_batch_size)
                tokens = self.valid_tokens
                counts = self.valid_counts
            else:
                indices = torch.split(torch.tensor(range(self.args.num_docs_test)), self.args.eval_batch_size)
                tokens = self.test_tokens
                counts = self.test_counts

            ## get \beta here
            beta, _, _ = m.get_beta()

            ### do dc and tc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(self.args.num_docs_test_1)), self.args.eval_batch_size)
            for idx, ind in enumerate(indices_1):
                ## get theta from first half of docs
                data_batch_1, target_batch_1 = get_batch(self.test_1_tokens, self.test_1_counts, ind, self.args.vocab_size, self.device,
                                                         self.test_targets)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if self.args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = m.get_theta(normalized_data_batch_1)

                ## get prediction loss using second half
                data_batch_2, target_batch_2 = get_batch(self.test_2_tokens, self.test_2_counts, ind, self.args.vocab_size, self.device,
                                                         self.test_targets)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)

                loss = recon_loss / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*' * 100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*' * 100)
            if tc or td:
                topk = 25  # word number per topic
                beta = beta.data.cpu().numpy()  # beta: 50x18000
                if tc:
                    print('Computing topic coherence...')
                    # cntm_eval
                    top_k_list = []
                    top_words_list = []
                    for k in range(len(beta)):
                        top_k = list(beta[k].argsort()[-topk:][::-1])
                        top_k_list.append(top_k)
                        top_words = [self.vocab[a] for a in top_k]
                        top_words_list.append(top_words)
                    calc_topic_coherence('CNTM', top_words_list, self.args.dataset)
                    # calc_tc('CNTM', top_k_list, self.train_tokens, self.args.dataset)
                if td:
                    print('Computing topic diversity...')
                    top_words_list = []
                    num_topics = beta.shape[0]
                    list_w = np.zeros((num_topics, topk))
                    for k in range(len(beta)):
                        idx = list(beta[k].argsort()[-topk:][::-1])
                        list_w[k, :] = idx
                        top_words = [self.vocab[a] for a in idx]
                        top_words_list.append(top_words)
                    calc_topic_diversity('CNTM', list_w, self.args.dataset, topk, 'TD')
                    calc_topic_diversity('CNTM', top_words_list, self.args.dataset, topk, 'TU')
            return ppl_dc

