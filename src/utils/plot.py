import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import util
from scipy import spatial
import random

tsne = TSNE(n_components=2, init='pca', random_state=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_topic_n_mean(name, v_list):
    scales = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total_num = len(v_list)
    for scale in scales:
        valid_num = int(total_num * scale)
        print('{} Mean with {}: {}'.format(name, scale, np.mean(v_list[:valid_num])))

def nearest_neighbors(word, embeddings, vocab, sim_num=10):
    vectors = embeddings.data.cpu().numpy()
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors_indices = mostSimilar[:sim_num]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors_indices]
    return nearest_neighbors, nearest_neighbors_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_word_points(beta, vocab, w_emb, topic_indices, type_num, topic_word_nu=10):
    print('Plot t-SNE embeddings, {} words per topic...'.format(topic_word_nu))

    top_words_all = []
    top_words_embedding = []
    topic_as_tags = []
    print('\n')

    plot_point = False
    if topic_word_nu > 10:
        plot_point = True

    for i, k in enumerate(topic_indices): # topic_indices:
        gamma = beta[k]

        # top prob words
        top_words_index = list(gamma.cpu().numpy().argsort()[-topic_word_nu:][::-1])
        top_words = [vocab[i] for i in top_words_index]
        if topic_word_nu <= 10:
            print('Topic {}: {}'.format(k, top_words))

        topic_as_tags.extend([i]*len(top_words))
        top_words_all.extend(top_words)
        top_words_embedding.extend([w_emb[i].tolist() for i in top_words_index])

    result = tsne.fit_transform(top_words_embedding)
    if plot_point is True:
        plot_embedding(result, topic_as_tags, type_num, '')
    else:
        plot_embedding(result, topic_as_tags, type_num, '', top_words_all)

def plot_embedding(data, targets, type_num, title='', labels=None):
    if targets is not None:
        sum_dict = {}
        for target in targets:
            if target not in sum_dict:
                sum_dict[target] = 1
            else:
                sum_dict[target] += 1
        top_targets = sorted(sum_dict, key=sum_dict.__getitem__, reverse=True)[0:type_num]
        filt_data = []
        filt_targets = []
        for i, target in enumerate(targets):
            if target not in top_targets:
                continue
            filt_data.append(data[i])
            filt_targets.append(top_targets.index(target))

        data = filt_data
        targets = filt_targets

    data = tsne.fit_transform(data)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # fig = plt.figure()
    # ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # for classify
        if targets is None:
            color_id = 0
        else:
            color_id = targets[i]

        if color_id >=0 and color_id < 9:
            color=plt.cm.Set1(color_id)
        else:
            color=plt.cm.Set2(color_id - 12)

        # show labels of points
        if labels is None:
            point = str(".")
        elif labels[i].startswith('Topic'):
            point = '*'
        else:
            point = labels[i]

        plt.text(data[i, 0], data[i, 1],
                 point, color=color,
                 fontdict={'weight': 'bold', 'size': 9})

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.xticks([-0.1, 1.2])
    plt.yticks([-0.1, 1.2])
    plt.title(title)
    plt.show()

def plot_embedding_specific(vocob, data, w_indices, targets, neighbors, title=''):
    data = tsne.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # fig = plt.figure()
    # ax = plt.subplot(111)
    for w_i in range(data.shape[0]):
        if w_indices is None:
            target = targets[w_i]
            point = neighbors[w_i]
        else:
            if w_i not in w_indices:
                continue
            target = targets[w_indices.index(w_i)]
            point = vocob[w_i]

        # for color
        color_id = target
        if color_id >=0 and color_id < 9:
            color=plt.cm.Set1(color_id)
        else:
            color=plt.cm.Set3(color_id - 9)
        plt.text(data[w_i, 0], data[w_i, 1],
                 point, color=color,
                 fontdict={'weight': 'bold', 'size': 9})

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.xticks([-0.1, 1.2])
    plt.yticks([-0.1, 1.2])
    plt.title(title)
    plt.show()

def plot_word_points(beta, vocab, w_emb, t_emb, type_num, topic_word_nu=10):
    print('Plot t-SNE embeddings, {} words per topic...'.format(topic_word_nu))

    top_words_all = []
    top_words_embedding = []
    topic_as_tags = []
    print('\n')

    plot_point = False
    if topic_word_nu > 10:
        plot_point = True

    # topic_indices = list(np.random.choice(type_num, min(10, type_num))) # 10 random topics
    topic_indices = random.sample(range(1, type_num), min(10, type_num))
    # topic_indices = [1, 2, 12, 20, 27, 28, 42, 45, 47, 49]
    # topic_indices = [12, 0, 5, 19, 9, 10, 15]
    for i, k in enumerate(topic_indices): # topic_indices:
        gamma = beta[k]
        topic_emb = t_emb[k].tolist()

        # top prob words
        top_words_index = list(gamma.cpu().numpy().argsort()[-topic_word_nu:][::-1])
        top_words = [vocab[i] for i in top_words_index]
        if topic_word_nu <= 10:
            print('Topic {}: {}'.format(k, top_words))

        topic_as_tags.extend([i]*(len(top_words) + 1))
        top_words.append('Topic ' + str(i))
        top_words_all.extend(top_words)
        ebds = [w_emb[i].tolist() for i in top_words_index]
        ebds.append(topic_emb)
        top_words_embedding.extend(ebds)

    if plot_point is True:
        plot_embedding(top_words_embedding, topic_as_tags, topic_word_nu)
    else:
        plot_embedding(top_words_embedding, topic_as_tags, topic_word_nu, '', top_words_all)

def get_pos_neg_bows(bows, docs_topics, topic_embeddings, targets=None):
    # bows: 1000x18000, docs_topics: 1000x50, topic_embeddings: 50x300, embedding_words: 18000x300
    pos_bows = []
    neg_bows = []
    real_doc_topic_embeddings = []

    # 1. build search tree with doc_topics
    # tree = spatial.KDTree(doc_topics.tolist())
    # for i, doc_topic in enumerate(doc_topics):
    #     closest = tree.query(doc_topic.tolist(), k=len(doc_topics))

    # 2. build search tree with doc_embeddings # 1000x300
    doc_embeddings = docs_topics@topic_embeddings
    batch_size = len(doc_embeddings)
    idx_permute = np.random.permutation(batch_size).astype(int)
    sub_batch_size = int(batch_size / 100)
    sub_indx_list = idx_permute[:sub_batch_size]
    sub_doc_embeddings = torch.stack([doc_embeddings[idx_permute[i]] for i in range(sub_batch_size)])
    tree = spatial.KDTree(sub_doc_embeddings.tolist())
    for i, (doc_embedding, doc_topic) in enumerate(zip(doc_embeddings, docs_topics)):
        closest = tree.query(doc_embedding.tolist(), k=sub_batch_size)

        if sub_indx_list[closest[1][0]] == i:
            pos_id = sub_indx_list[closest[1][1]]
        else:
            pos_id = sub_indx_list[closest[1][0]]
        neg_id = sub_indx_list[closest[1][-1]]
        pos_bows.append(bows[pos_id])
        neg_bows.append(bows[neg_id])
        # # get a positive/negative bow to the current bow: supervised
        # for j in range(len(bows)):
        #     doc_idx = closest[1][j]
        #     if i == doc_idx: # self
        #         continue
        #     if targets is None: # unsupervised
        #         pos_bows.append(bows[doc_idx])
        #         break
        #     curr_bow_target = targets[i]
        #     if targets[doc_idx] == curr_bow_target:
        #         pos_bows.append(bows[doc_idx])
        #         break
        # for k in range(len(bows)):
        #     doc_idx = closest[1][-k - 1]
        #     if targets is None:
        #         neg_bows.append(bows[doc_idx])
        #         break
        #     curr_bow_target = targets[i]
        #     if targets[doc_idx] != curr_bow_target:
        #         neg_bows.append(bows[doc_idx])
        #         break

        # get real topic embedding of each doc
        doc_max_topic_prob_i = list(doc_topic).index(max(doc_topic))
        real_doc_topic_embeddings.append(topic_embeddings[doc_max_topic_prob_i])

    # 3. use cosine-similarity and torch.topk to find the closest
    # 0.6s
    # doc_embeddings # 1000x300
    # https://www.sbert.net/examples/applications/semantic-search/README.html
    # doc_embeddings = docs_topics @ topic_embeddings
    # doc_embeddings = doc_embeddings.to(device)
    # batch_size = len(doc_embeddings)
    # idx_permute = np.random.permutation(batch_size).astype(int)
    # sub_batch_size = int(batch_size / 100 + 1)
    # sub_indx_list = idx_permute[:sub_batch_size]
    # sub_doc_embeddings = torch.stack([doc_embeddings[idx_permute[i]] for i in range(sub_batch_size)]).to(device)
    # for i, (doc_embedding, doc_topic) in enumerate(zip(doc_embeddings, docs_topics)):
    #     cos_scores = util.dot_score(doc_embedding, sub_doc_embeddings)[0]
    #     top_results = torch.topk(cos_scores, k=sub_batch_size)
    #
    #     if sub_indx_list[top_results[1][0]] == i:
    #         pos_id = sub_indx_list[top_results[1][1]]
    #     else:
    #         pos_id = sub_indx_list[top_results[1][0]]
    #     neg_id = sub_indx_list[top_results[1][-1]]
    #     pos_bows.append(bows[pos_id])
    #     neg_bows.append(bows[neg_id])
    # for doc_topic in docs_topics:
    #     # get real topic embedding of each doc
    #     doc_max_topic_prob_i = list(doc_topic).index(max(doc_topic))
    #     real_doc_topic_embeddings.append(topic_embeddings[doc_max_topic_prob_i])

    # 4. use cosine-similarity and semantic_search to find the closest
    # https://www.sbert.net/examples/applications/semantic-search/README.html
    # 0.91s
    # doc_embeddings = docs_topics @ topic_embeddings
    # doc_embeddings = doc_embeddings.to(device)
    # hits = util.semantic_search(doc_embeddings, doc_embeddings, score_function=util.dot_score, top_k=len(bows))
    # for i, hit in enumerate(hits):
    #     if hit[0]['corpus_id'] == i:
    #         pos_id = hit[1]['corpus_id']
    #     else:
    #         pos_id = hit[0]['corpus_id']
    #     neg_id = hit[-1]['corpus_id']
    #     pos_bows.append(bows[pos_id])
    #     neg_bows.append(bows[neg_id])
    # for doc_topic in docs_topics:
    #     # get real topic embedding of each doc
    #     doc_max_topic_prob_i = list(doc_topic).index(max(doc_topic))
    #     real_doc_topic_embeddings.append(topic_embeddings[doc_max_topic_prob_i])

    # return
    if len(pos_bows) == 0 or len(neg_bows) == 0 or len(real_doc_topic_embeddings) == 0:
        return None, None, None
    return torch.stack(pos_bows), torch.stack(neg_bows), torch.stack(real_doc_topic_embeddings)

def bow_to_docs_embeddings(bows, word_embeddings):
    docs_embs = []

    for doc in bows:
        word_index_list = torch.nonzero(doc).squeeze()
        word_embedding_list_per_doc = []
        for w_i in word_index_list:
            word_embedding_list_per_doc.append(word_embeddings[w_i.item()])
        docs_embs.append(word_embedding_list_per_doc)

    return docs_embs

def docs_sim_topics(docs, topics):
    docs_topics_relevance = []
    docs_most_sim_topic = []

    for doc in docs:
        doc_sim_topics = [getTopicRelevance(topic, doc) for topic in topics]
        max_sim = max(doc_sim_topics)
        max_sim_topic_i = doc_sim_topics.index(max_sim)
        docs_topics_relevance.append(doc_sim_topics)
        docs_most_sim_topic.append(max_sim_topic_i)

    return np.array(docs_topics_relevance), np.array(docs_most_sim_topic)

def getTopicRelevance(topic, words):
    # sim_list = [np.square(cosSim(np.array(topic.data.to("cpu")), np.array(word.data.to("cpu")))) for word in words]
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    sim_list = [torch.square(cos(topic, word)) for word in words]
    return sum(sim_list)/len(sim_list)

def cosSim(array1, array2):
    if (np.linalg.norm(array1) * np.linalg.norm(array2)) == 0:
        return 0
    return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))
