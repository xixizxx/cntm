import requests
import numpy as np
import codecs
import json
import os
import sys
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.feature_extraction.text import CountVectorizer

output_dir = os.path.dirname(__file__) + '/metrics/'
os.makedirs(output_dir, exist_ok=True)

portions = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

IP = '10.214.199.136'

def calc_topic_diversity(alg, top_words_list, dataset, total_topk=25, type='TU'):
    result = {}
    num_topics = len(top_words_list)
    for portion in portions:
        valid_num = int(total_topk * portion)
        top_words_list_portion = [top_words[:valid_num] for top_words in top_words_list]
        if type == 'TD':
            n_unique = len(np.unique(top_words_list_portion))
            topk = len(top_words_list_portion[0])
            TD = n_unique / (topk * num_topics)
            result[str(portion)] = TD
        else:  # 'TU'
            texts = [' '.join(top_words) for top_words in top_words_list_portion]
            K = len(texts)
            T = len(texts[0].split())
            vectorizer = CountVectorizer()
            counter = vectorizer.fit_transform(texts).toarray()
            TU = 0.0
            TF = counter.sum(axis=0)
            cnt = TF * (counter > 0)
            for i in range(K):
                TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
            TU /= K
            result[str(portion)] = TU

    print('{}-{}-K{}-TD is {}'.format(alg, dataset, num_topics, np.mean(list(result.values()))))
    output_file = '{}/{}-{}-{}-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), type, num_topics)
    fout = codecs.open(output_file, "w", 'utf-8')
    json_str = json.dumps(result)
    fout.write(f'{json_str}\n')
    fout.close()

def calc_tc(alg, top_k_list, data, dataset):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(top_k_list)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = top_k_list[k][:11]  # original paper setting 11
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    # print('Topic coherence is: {}'.format(TC))

    print('{}-{}-K{}-TC is {}'.format(alg, dataset, num_topics, TC))
    output_file = '{}/{}-{}-TC-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), num_topics)
    fout = codecs.open(output_file, "w", 'utf-8')
    json_str = json.dumps({'TC':TC})
    fout.write(f'{json_str}\n')
    fout.close()

def calc_topic_coherence(alg, top_words_list, dataset):
    num_topics = len(top_words_list)
    TC_cp = []
    TC_ca = []
    TC_npmi = []
    TC_uci = []
    TC_umass = []
    for top_words in top_words_list:
        # response = requests.get('http://' + IP + ':7777/service/cp?words=' + '%20'.join(top_words[:10]))
        # TC_cp.append(float(response.text))
        # response = requests.get('http://' + IP + ':7777/service/ca?words=' + '%20'.join(top_words[:10]))
        # TC_ca.append(float(response.text))
        response = requests.get('http://' + IP + ':7777/service/npmi?words=' + '%20'.join(top_words[:10]))
        TC_npmi.append(float(response.text))
        # response = requests.get('http://' + IP + ':7777/service/uci?words=' + '%20'.join(top_words[:10]))
        # TC_uci.append(float(response.text))
        # response = requests.get('http://' + IP + ':7777/service/umass?words=' + '%20'.join(top_words[:10]))
        # TC_umass.append(float(response.text))

    # write_data('{}/{}-{}-CP-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), num_topics), np.sort(TC_cp)[::-1])
    # write_data('{}/{}-{}-CA-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), num_topics), np.sort(TC_ca)[::-1])
    # write_data('{}/{}-{}-UMASS-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), num_topics), np.sort(TC_umass)[::-1])
    # write_data('{}/{}-{}-UCI-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), num_topics), np.sort(TC_uci)[::-1])
    write_data('{}/{}-{}-NPMI-K{}.json'.format(output_dir, alg, fix_ds_name(dataset), num_topics), np.sort(TC_npmi)[::-1])

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l]
            if len(doc.shape) == 1:
                doc = np.array([doc])
            doc = doc.squeeze(0)
            if doc.size == 1:
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l]
        if len(doc.shape) == 1:
            doc = np.array([doc])
        doc = doc.squeeze(0)
        if doc.size == 1:
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj

def write_data(output_file, sorted_m_list):
    fout = codecs.open(output_file, "w", 'utf-8')
    result = {}
    total_num = len(sorted_m_list)
    for scale in portions:
        valid_num = int(total_num * scale)
        avg_metric_per_scale = np.mean(sorted_m_list[:valid_num])  # avg on top 20% coherent values
        print('{}: Mean of scale{} is {}'.format(output_file, scale, np.mean(sorted_m_list[:valid_num])))
        result[scale] = avg_metric_per_scale

    json_str = json.dumps(result)
    fout.write(f'{json_str}\n')
    fout.close()

def fix_ds_name(dataset):
    if dataset == '20News':
        return '20ng'
    if dataset == 'TMN':
        return 'tmn'
    if dataset == 'Webs':
        return 'webs'
    return dataset

def avg_score_on_Ks(alg='CNTM', dataset='20ng', metric='NPMI', portion=None):
    K_settings = [20, 30, 50, 75, 100]
    max = None
    min = None
    sum = 0
    count = 0
    for k in K_settings:
        file = os.path.dirname(__file__) + '/metrics/' + alg + '-' + fix_ds_name(dataset) + '-' + metric + '-K' + str(k) + '.json'
        if os.path.exists(file) is False:
            continue
        with open(file, encoding='utf-8') as f:
            for line in f:
                jsonStr = json.loads(line.rstrip())
                if portion is None:
                    v = jsonStr[metric]
                else:
                    v = jsonStr[str(portion)] if str(portion) in jsonStr else None
                if v is None:
                    continue
                max = v if max is None or v > max else max
                min = v if min is None or v < min else min
                sum += v
            f.close()
        count += 1
    avg = sum / count if count > 0 else 0

    return avg, max, min

def plot(dataset, metric):
    print("====================== plot results ========================")
    algs = ['CNTM', 'NSTM', 'DVAE', 'WLDA', 'ETM', 'LDA', 'ProdLDA']
    # portions = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    portions = [0.2, 0.4, 0.6, 0.8, 1.0]

    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)

    # scores of the same alg on different datasets
    # scores of the same alg on different portions

    scores = []
    spot_sets = []
    for alg in algs:
        file = os.path.dirname(__file__) + '/../../../metrics-avg/' + alg + '-' + fix_ds_name(dataset) + '-' + metric + '.json'
        if os.path.exists(file) is False:
            continue

        with open(file, encoding='utf-8') as f:
            for line in f:
                jsonStr = json.loads(line.rstrip())
                break

            scores_per_alg = []
            alg_spots = DataFrame(columns=['x', 'y'])
            if metric in ['TC']:
                for portion in portions:
                    avg = jsonStr[str(portion)]
                    scores_per_alg.append(avg)
                    alg_spots = alg_spots.append(Series([portion, avg], index=['x', 'y']), ignore_index=True)
            else:
                for portion in portions:
                    v = jsonStr[str(portion)]
                    avg = v['avg'] if type(v) is dict else v
                    scores_per_alg.append(avg)
                    alg_spots = alg_spots.append(Series([portion, avg], index=['x', 'y']), ignore_index=True)
            scores.append(scores_per_alg)
            spot_sets.append(alg_spots)

    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)

    subplot.set_xlim(0.2, 1.0)
    if metric in ['TD', 'TU']:
        subplot.set_ylim(-0.1, 1.0)
    else:
        subplot.set_ylim(-0.15, 0.2)
        if dataset == 'tmn':
            subplot.set_ylim(-0.15, 0.25)
    subplot.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.xlabel("Proportion of the selected topics from 20% to 100%.")
    plt.ylabel("Avarage Scores (" + metric + ") on " + dataset)

    for i, alg_spots in enumerate(spot_sets):
        label = algs[i]
        marker = 'p'
        style = '-'
        if i == 0:
            color = 'green'
        elif i == 1:
            color = 'red'
        elif i == 2:
            color = 'yellow'
        elif i == 3:
            color = 'black'
        elif i == 4:
            color = 'magenta'
        elif i == 5:
            color = 'purple'
        elif i == 6:
            color = 'blue'
        else:
            continue

        subplot.plot(alg_spots.x, alg_spots.y, marker=marker, color=color, linestyle=style, label=label)

        # 设置label位置
        subplot.legend(loc=7)

    plt.legend()
    plt.show()

    # portions_size = len(portions)
    # alg_num = len(algs)
    # x = np.arange(portions_size)
    # total_width, n = 0.8, alg_num
    # width = total_width / n
    # x = x - (total_width - width) / 2
    #
    # for i, score_per_alg in enumerate(scores):
    #     plt.bar(x + i*width, score_per_alg,  width=width, label=algs[i], tick_label=portions, align="center")
    #
    # ax = plt.gca()
    # if metric in ['TD', 'TU']:
    #     ax.set_ylim([0.1, 1.0])
    # else:
    #     ax.set_ylim([-0.15, 0.2])
    # plt.legend(loc='upper center')
    # plt.show()

# Main
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2 or len(argv) > 3 :
        print('Usage: python ' + sys.argv[0] + ' 20ng CNTM')
        print('Usage: python ' + sys.argv[0] + ' 20ng')
        exit(0)
    dataset = sys.argv[1]

    metrics = ['CA', 'CP', 'NPMI', 'UCI', 'UMASS', 'TC', 'TU', 'TD']

    if len(argv) == 3:  # print and save plot data
        alg = sys.argv[2]
        for metric in metrics:
            outfile = os.path.dirname(__file__) + '/../../../metrics-avg/' + alg + '-' + fix_ds_name(dataset) + '-' + metric + '.json'
            fout = codecs.open(outfile, "w", 'utf-8')
            if metric == 'TC':
                avg, max, min = avg_score_on_Ks(alg, dataset, metric)
                print('{}-{}-{}: Avg Score={}, Max Score={}, Min Score={}'.format(alg, fix_ds_name(dataset), metric, avg, max, min))
                result = {'avg': avg, 'max': max, 'min': min}
            else:
                result = {}
                for portion in portions:
                    avg, max, min = avg_score_on_Ks(alg, fix_ds_name(dataset), metric, portion)
                    print('{}-{}-{}-{}: Avg Score={}, Max Score={}, Min Score={}'.format(alg, fix_ds_name(dataset), metric, portion, avg, max, min))
                    result[str(portion)] = {'avg': avg, 'max': max, 'min': min}
            json_str = json.dumps(result)
            fout.write(f'{json_str}\n')
            fout.close()
    else:  # plot
        for metric in ['NPMI', 'TD']:
            plot(dataset, metric)

        top_words_list = []
        topic_file = os.path.expanduser('~/dev/neural-topic-model/CLNTM/outputs/'+ dataset +'_50_25/topics.txt')
        with open(topic_file, 'r') as f:
            for line in f:
                word_list = line.split(' ')
                top_words_list.append(word_list)
            f.close()
        calc_topic_coherence('CLNTM', top_words_list, dataset)
        calc_topic_diversity('CLNTM', top_words_list, dataset)

    sys.exit(0)