import multiprocessing
import os
import sys
from collections import defaultdict
import pickle
import time
import copy
import re

import numpy as np
from sklearn.metrics import classification_report
from src.clustering import clustering

from src.build_data import group_data, slice_data, BATCH_SIZE, replace_pronoun
from src.preprocess import SPEAKER_MAP
FULL_OUTPUT = True

class Evaluator(object):
    def __init__(self, model, test_data_q):
        self.model = model
        self.test_data_q = test_data_q

    def fast_eval(self, test_data=False):
        Y_true = []
        Y_pred = []
        for data in self.test_data_q:
            if len(data) == 3:
                data = data[:2]
            for X, y in group_data(data, 40, BATCH_SIZE):
                pred = self.model.predict_on_batch(X)
                Y_true.append(y)
                Y_pred.append(pred)

        Y_true = np.concatenate(Y_true)
        Y_true = Y_true.flatten()
        Y_pred = np.concatenate(Y_pred)
        Y_pred = Y_pred.flatten()
        Y_pred = np.round(Y_pred).astype(int)

        return classification_report(Y_true, Y_pred, digits=3)

    def write_results(self, df, dest_path):
        n_files = len(self.test_data_q)
        print("# files: %d" % n_files)
        for i, data in enumerate(self.test_data_q):
            if i == n_files:
                break
            X, y, index_map = data
            pred = []
            for X, y in group_data([X, y], 40, batch_size=BATCH_SIZE):
                pred.append(self.model.predict_on_batch(X))
            pred = np.concatenate(pred)
            pred = pred.flatten()

            pair_results = {}
            for key in index_map:
                pair_results[(key[1], key[2])] = pred[index_map[key]]
            locs, clusters, _ = clustering(pair_results)
            doc_id = key[0]
            length = len(df.loc[df.doc_id == doc_id])

            sys.stdout.write("Saving results %d / %d\r" % (i + 1, n_files))
            sys.stdout.flush()
            corefs = ['-' for i in range(length)]
            for loc, cluster in zip(locs, clusters):
                start, end = loc
                if corefs[start] == '-':
                    corefs[start] = '(' + str(cluster)
                else:
                    corefs[start] += '|(' + str(cluster)

                if corefs[end] == '-':
                    corefs[end] = str(cluster) + ')'
                elif start == end:
                    corefs[end] += ')'
                else:
                    corefs[end] += '|' + str(cluster) + ')'
            with open(os.path.join(dest_path, doc_id.split('/')[-1]), 'w') as f:
                f.write('#begin document (%s);\n' % doc_id)
                for coref in corefs:
                    f.write(doc_id + '\t' + coref +'\n')
                f.write('\n#end document\n')
        print("Completed saving results!")


class TriadEvaluator(object):
    def __init__(self, model, test_input_gen, data_maker=slice_data, group_size=100):
        self.model = model
        self.test_input_gen = test_input_gen
        self.data_q_store = multiprocessing.Queue(maxsize=5)
        self.data_maker = data_maker
        self.group_size = group_size

    def fast_eval(self):
        """Fast evaluation from a subset of test files
           Scores are based on pairs only
        """
        # assert self.data_available
        Y_true = []
        Y_pred = []
        test_data_q = next(self.test_input_gen)
        for data in test_data_q:
            if len(data) == 3:
                data = data[:2]
            for X, y in self.data_maker(data, self.group_size, batch_size=10):
                if y.shape[-1] == 3:
                    y = y[:, 1:] if len(y.shape) == 2 else y[:, :, 1:]
                pred = self.model.predict(X) # (group_size, 3)
                Y_true.append(y)
                Y_pred.append(pred)

        Y_true = np.concatenate(Y_true)
        Y_true = Y_true.flatten()
        Y_pred = np.concatenate(Y_pred)
        Y_pred = Y_pred.flatten()
        Y_pred = np.round(Y_pred).astype(int)

        return classification_report(Y_true, Y_pred, digits=3)

    def write_results(self, df, dest_path, n_iterations, save_dendrograms=True, clustering_only=False, compute_linkage=False):
        """Perform evaluation on all test data, write results"""
        # assert self.data_available
        print("# files: %d" % n_iterations)

        all_pairs_true = []
        all_pairs_pred = []
        processed_docs = set([])
        doc_ids = df.doc_id.unique()
        i = n_iterations
        t = 3.6
        # t = 3.4
        method = 'average'
        criterion = 'distance'
        print("clustering parameters: t={}, method={}, criterion={}".format(t, method, criterion))

        while i > 0:
            if not clustering_only:
                test_data_q = next(self.test_input_gen)
                assert len(test_data_q) == 1  # only process one file
                if not test_data_q[0]:
                    i -= 1
                    continue
                X, y, index_map = test_data_q[0]
                if y.shape[-1] == 3:
                    y = y[:, 1:] if len(y.shape) == 2 else y[:, :, 1:]

                doc_id = list(index_map.keys())[0][0]  # python3 does not support keys() as a list


                if doc_id in processed_docs:
                    # print("%s already processed before!" % doc_id)
                    time.sleep(10)
                    continue
                processed_docs.add(doc_id)
                pred = []
                for X, _ in self.data_maker([X, y], self.group_size, batch_size=1):  # batch size 1 to avoid random orders of parts
                    pred.append(self.model.predict(X))
                pred = np.concatenate(pred)
                pred = np.reshape(pred, [-1, 2])  # in case there are batches

                true = np.reshape(y, [-1, 2])

                pair_results = defaultdict(list)
                pair_true = {}
                for key in index_map:
                    assert key[1][0] <= key[2][0] <= key[3][0]
                    pair_results[(key[2], key[3])].append(pred[index_map[key]][0])  # result of (b, c)
                    if key[1] != key[2]:  # handle extra triads at beginning of file
                        pair_results[(key[1], key[3])].append(pred[index_map[key]][1])  # result of (a, c)

                    pair_true[(key[2], key[3])] = true[index_map[key]][0]
                    pair_true[(key[1], key[3])] = true[index_map[key]][1]

                # save raw scores
                pickle.dump(pair_results, open(os.path.join(dest_path, 'raw_scores', doc_id.split('/')[-1]+'results.pkl'), 'wb'))

                doc_df = df.loc[df.doc_id == doc_id]
                doc_df = doc_df.reset_index()
                original_doc_df = copy.copy(doc_df)
                replace_pronoun(doc_df)
                speakers = [SPEAKER_MAP.get(s, '-') for s in doc_df.speaker.unique()]

                pair_results_mean = {}
                for key, value in pair_results.items():
                    if doc_df.iloc[key[0][0]].word == doc_df.iloc[key[1][0]].word and doc_df.iloc[key[1][0]].word in speakers:
                        mean_value = 1.0  # maximum value
                    else:
                        # mean_value = TriadEvaluator.top_n_mean(value, 0)
                        mean_value = TriadEvaluator.top_n_mean(value, 3)
                    pair_results_mean[key] = mean_value
                    all_pairs_pred.append(mean_value)
                    all_pairs_true.append(pair_true[key])

                locs, clusters, linkage = clustering(pair_results_mean, binarize=False, t=t, method=method)
                _, clusters_true, linkage_true = clustering(pair_true, binarize=False, t=t, method=method)

                clusters = TriadEvaluator.remove_singletons(clusters)

            else:  # clustering only
                from scipy.cluster.hierarchy import inconsistent, fcluster
                doc_id = doc_ids[i - 1]

                doc_df = df.loc[df.doc_id == doc_id]
                doc_df = doc_df.reset_index()
                original_doc_df = copy.copy(doc_df)
                replace_pronoun(doc_df)
                speakers = [SPEAKER_MAP.get(s, '-') for s in doc_df.speaker.unique()]

                if doc_id.split('/')[-1] in ('wsj_2390', 'wsj_2390-0', 'cnn_0008-8'):
                    print("file not found:", doc_id)
                    i -= 1
                    continue

                if compute_linkage:
                    pair_results = pickle.load(open(os.path.join(dest_path, 'raw_scores', doc_id.split('/')[-1]+'results.pkl'), 'rb'))

                    pair_results_mean = {}
                    for key, value in pair_results.items():
                        if doc_df.iloc[key[1][0]].word in (doc_df.iloc[key[0][0]].word, doc_df.iloc[key[0][1]].word) and doc_df.iloc[key[1][0]].word in speakers:
                            mean_value = 1.0  # maximum value
                        else:
                            mean_value = TriadEvaluator.top_n_mean(value, 0)
                        pair_results_mean[key] = mean_value

                    pair_results_mean = TriadEvaluator.attach_proper_names(pair_results_mean, doc_df)
                    locs, clusters, linkage = clustering(pair_results_mean, binarize=False, t=t, method=method)
                    locs, clusters, linkage = TriadEvaluator.pick_antecedent(pair_results_mean, doc_df, locs, clusters, linkage, t, method)


                else:
                    save_dendrograms = False
                    linkage = np.load(os.path.join(dest_path, 'linkages', doc_id.split('/')[-1] + '.npy'))
                    with open(os.path.join(dest_path, 'linkages', doc_id.split('/')[-1]+'-locs.pkl'), 'rb') as f:
                        locs = pickle.load(f)

                    clusters = fcluster(linkage, criterion=criterion, depth=2, R=None, t=t)

                clusters = TriadEvaluator.remove_singletons(clusters)

            if save_dendrograms:
                np.save(os.path.join(dest_path, 'linkages', doc_id.split('/')[-1] + '.npy'), linkage)
                with open(os.path.join(dest_path, 'linkages', doc_id.split('/')[-1] + '-locs.pkl'), 'wb') as f:
                    pickle.dump(locs, f)
                if not compute_linkage:
                    np.save(os.path.join(dest_path, 'true-linkages', doc_id.split('/')[-1] + '.npy'), linkage_true)

            length = len(doc_df)
            # print("Saving %s results..." % doc_id)
            sys.stdout.write("Saving results %d / %d\r" % (n_iterations - i + 1, n_iterations))
            sys.stdout.flush()
            corefs = ['-' for _ in range(length)]
            for loc, cluster in zip(locs, clusters):

                if cluster == -1:  # singletons
                    continue

                start, end = loc
                if corefs[start] == '-':
                    corefs[start] = '(' + str(cluster)
                else:
                    corefs[start] += '|(' + str(cluster)

                if corefs[end] == '-':
                    corefs[end] = str(cluster) + ')'
                elif start == end:
                    corefs[end] += ')'
                else:
                    corefs[end] += '|' + str(cluster) + ')'
            fn = doc_id.split('/')[-1]
            if FULL_OUTPUT:
                fn += '.corenlp_conll'
                file_id = doc_df.iloc[0].file_id
                part_n = doc_id.split('-')[-1]
                if len(part_n) == 1:
                    part_n = '00' + part_n
                elif len(part_n) == 2:
                    part_n = '0' + part_n
                else:
                    print(part_n)
                    raise ValueError
            with open(os.path.join(dest_path, 'responses', fn), 'w') as f:
                f.write('#begin document (%s); part %s\n' % (file_id, part_n))
                for loc, coref in enumerate(corefs):
                    if FULL_OUTPUT:
                        columns = [str(s) for s in original_doc_df.iloc[loc].values[:-1]][2:]
                        f.write('\t'.join(columns) + '\t' + coref +'\n')
                    else:
                        word = doc_df.iloc[loc].word
                        f.write(doc_id + '\t' + word + '\t' + coref + '\n')
                f.write('\n#end document\n')

            i -= 1

        print("Completed saving results!")
        if not clustering_only:
            print("Pairwise evaluation:")
            print("True histogram", np.histogram(all_pairs_true, bins=4))
            print("Prediction histogram", np.histogram(all_pairs_pred, bins=4))
            print(classification_report(all_pairs_true, np.round(all_pairs_pred), digits=3))

    @staticmethod
    def last_n_values(values, n):
        if len(values) >= n:
            value = np.mean(values[-n:])
        else:
            value = np.mean(values)
        return value

    @staticmethod
    def top_n_mean(values, n):
        if n >= 1:
            values.sort(reverse=True)
            if len(values) >= n:
                values = values[:n]
        return np.mean(values)

    @staticmethod
    def median(values):
        values.sort(reverse=True)
        return values[len(values)/2]

    @staticmethod
    def nonlinear_mean(values):
        return np.mean(np.round(values))

    @staticmethod
    def bottom_n_mean(values, n):
        if n >= 1:
            values.sort()
            if len(values) >= n:
                values = values[:n]
        return np.mean(values)

    @staticmethod
    def remove_singletons(clusters):
        counter = defaultdict(int)
        for item in clusters:
            counter[item] += 1
        for i, item in enumerate(clusters):
            if counter[item] == 1:
                clusters[i] = -1  # use -1 as a special value for singletons
        return clusters

    @staticmethod
    def attach_singletons(linkage, t=3.5):

        n = len(linkage) + 1
        clusters = np.where(linkage[:, 2] > t)[0]
        mentions = np.where(linkage[:, 0] < n)[0]
        singletons = set(clusters).intersection(set(mentions))
        for singleton in singletons:
            linkage[singleton, 2] = t - 0.1
        print("singletons attached", singletons)

    @staticmethod
    def remove_embeds(locs, clusters):
        n = len(locs)
        for i, loc0 in enumerate(locs):
            for j in range(i + 1, n):
                loc1 = locs[j]
                if loc0[0] == loc0[1] or loc1[0] == loc1[1]:  #  single words likely to be pronouns. preserve them.
                    continue
                if clusters[i] == clusters[j]:
                    if loc0[0] <= loc1[0] and loc0[1] >= loc1[1]:
                        clusters[j] = -1  # -1 will not be written in result files
                    elif loc0[0] >= loc1[0] and loc0[1] <= loc1[1]:
                        clusters[i] = -1

        return clusters

    @staticmethod
    def pick_antecedent(pair_results, doc_df, locs, clusters, linkage, t, method, iters=0):
        pronouns = set(['he', 'she', 'it', 'they', 'his', 'her', 'their', 'him', 'them', 'hers', 'its', 'theirs'])

        def are_all_pronous(locs):
            for loc in locs:
                if loc[0] != loc[1] or doc_df.iloc[loc[0]].word.lower() not in pronouns:
                    return False
            return True

        locs = [tuple(loc) for loc in locs]  # convert list to tuple
        loc2cluster = dict(zip(locs, clusters))
        cluster2locs = defaultdict(list)
        for loc, cluster in loc2cluster.items():
            cluster2locs[cluster].append(loc)

        all_pair_results = {}
        for k, v in pair_results.items():
            if k[0][0] <= k[1][0]:
                all_pair_results[k] = v

        leading_pronouns = []
        for cluster, items in cluster2locs.items():
            if are_all_pronous(items):
                leading_pronouns.append(items[0])
        if not leading_pronouns:
            assert len(locs) == len(clusters)
            return locs, clusters, linkage

        for loc in leading_pronouns:
            pre_candidates = [(key, all_pair_results[key]) for key in all_pair_results if key[1] == loc \
                                and key[0][0] !=  key[1][0]]
            post_candidates = [(key, all_pair_results[key]) for key in all_pair_results if key[0] == loc \
                               and (key[1][0] != key[1][1] or doc_df.iloc[key[1][0]].word.lower() not in pronouns) \
                               and key[0][0] !=  key[1][0]]
            if len(pre_candidates) > 6:
                pre_candidates = sorted(pre_candidates)[-6:]
            if len(post_candidates) > 2:
                post_candidates = sorted(post_candidates)[:2]
            candidates = pre_candidates + post_candidates

            if not candidates:
                continue
            match = sorted(candidates, key=lambda x: x[1])[-1][0]  # loc
            cluster = loc2cluster[loc]
            other_locs = cluster2locs[cluster]
            if loc == match[0]:
                antecedent = match[1]
            else:
                antecedent = match[0]
            if iters < 4:
                for item in other_locs:
                    pair_results[item, antecedent] = 1.0
                    pair_results[antecedent, item] = 1.0
            else:
                antecedent_cluster = loc2cluster[antecedent]
                antecedent_locs = cluster2locs[antecedent_cluster]
                for loc1 in other_locs:
                    for loc2 in antecedent_locs:
                        pair_results[loc1, loc2] = 1.0
                        pair_results[loc2, loc1] = 1.0

        # print('%d recursion(s) finished in' %(iters + 1), doc_df.iloc[0].doc_id)

        locs, clusters, linkage = clustering(pair_results, binarize=False, t=t, method=method)
        return TriadEvaluator.pick_antecedent(pair_results, doc_df, locs, clusters, linkage, t, method, iters=iters+1)

    @staticmethod
    def adjust_cluster_distances(all_pair_results, locs, clusters, t, doc_id):
        def recompute_linkage(locs1, locs2):
            accumulate = 0
            count = 0
            for loc1 in locs1:
                for loc2 in locs2:
                    assert loc1 != loc2
                    if (loc1, loc2) in all_pair_results:
                        score = all_pair_results[(loc1, loc2)]
                    elif (loc1, loc2) in all_pair_results:
                        score = all_pair_results[(loc2, loc1)]
                    else:
                        continue
                    accumulate += min(10, 1/score)
                    count += 1
            if count == 0:
                return t + 0.2
            return accumulate / count

        original_clusters = copy.copy(clusters)
        locs = [tuple(loc) for loc in locs]  # convert list to tuple
        loc2cluster = dict(zip(locs, clusters))
        cluster2locs = defaultdict(list)
        for loc, cluster in loc2cluster.items():
            cluster2locs[cluster].append(loc)

        clusters = list(cluster2locs.keys())  # so no duplicate
        n_clusters = len(clusters)
        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                if clusters[i] in cluster2locs and clusters[j] in cluster2locs:
                    locs1 = cluster2locs[clusters[i]]
                    locs2 = cluster2locs[clusters[j]]
                    adjusted_distance = recompute_linkage(locs1, locs2)
                    if adjusted_distance < t:
                        for loc in locs2:
                            loc2cluster[loc] = clusters[i]
                        cluster2locs[clusters[i]] += cluster2locs[clusters[j]]
                        del cluster2locs[clusters[j]]

        new_clusters = [loc2cluster[loc] for loc in locs]
        merged = len(set(original_clusters)) - len(set(new_clusters))
        if merged:
            print("%d out of %d clusters merged in %s" % (merged + 1, len(set(original_clusters)), doc_id))

        return new_clusters

    @staticmethod
    def attach_proper_names(pair_results, doc_df):

        def connected_components(lists):
            neighbors = defaultdict(set)
            seen = set()
            for each in lists:
                for item in each:
                    neighbors[item].update(each)

            def component(node, neighbors=neighbors, seen=seen, see=seen.add):
                nodes = set([node])
                next_node = nodes.pop
                while nodes:
                    node = next_node()
                    see(node)
                    nodes |= neighbors[node] - seen
                    yield node

            for node in neighbors:
                if node not in seen:
                    yield sorted(component(node))

        all_pair_results = copy.copy(pair_results)
        all_locs = set([])
        for key in all_pair_results:
            all_locs.add(key[0])
            all_locs.add(key[1])

        for key0 in all_locs:
            for key1 in all_locs:
                if key0 == key1:
                    continue
                if (key0, key1) in all_pair_results:
                    all_pair_results[key1, key0] = all_pair_results[key0, key1]
                elif (key1, key0) in all_pair_results:
                    all_pair_results[key0, key1] = all_pair_results[key1, key0]
                else:
                    all_pair_results[key0, key1] = 0.0  # just to get the pairs
                    all_pair_results[key1, key0] = 0.0

        identicals = defaultdict(set)
        for pair in all_pair_results:
            e1, e2 = pair
            if e1[1] - e1[0] > 1 or e2[1] - e2[0] > 1:
                continue

            if doc_df.iloc[e1[0]].word == doc_df.iloc[e2[0]].word and re.search('(PERSON)|(ORG)', doc_df.iloc[e1[0]].name_entities):
                if e1[1] - e1[0] == e2[1] - e2[0] == 1 or (e1[1] - e1[0] == e2[1] - e2[0] == 2 and doc_df.iloc[e1[1]].word == doc_df.iloc[e2[1]].word):
                    all_pair_results[pair] = 1.0
                    all_pair_results[pair[1], pair[0]] = 1.0
                    words = ''
                    for i in range(e1[1] - e1[0] + 1):
                        words += doc_df.iloc[e1[i]].word
                    identicals[words].update([e1, e2])
                elif e1[1] - e1[0] + e2[1] - e2[0] > 2 and 'POS' in (doc_df.iloc[e1[1]].pos, doc_df.iloc[e2[1]].pos):
                    if e1[1] - e1[0] < e2[1] - e2[0]:
                        e = e1
                    else:
                        e = e2
                    words = ''
                    for i in range(e[0], e[1] + 1):
                        words += doc_df.iloc[i].word
                    identicals[words].update([e1, e2])

            elif doc_df.iloc[e1[0]].word == doc_df.iloc[e2[0]].word and all_pair_results[pair] > 0.97:
                if e1[1] - e1[0] > e2[1] - e2[0]:
                    e = e1
                else:
                    e = e2
                words = ''
                for i in range(e[0], e[1] + 1):
                    words += doc_df.iloc[i].word
                identicals[words].update([e1, e2])

        # print(identicals)
        groups = [tuple(sorted(list(v))) for k, v in identicals.items()]

        # reduced_groups = list(connected_components(groups))
        reduced_groups = connected_components(groups)
        for locs in reduced_groups:
            n = len(locs)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # print(locs[i], locs[j], all_pair_results[locs[i], locs[j]])
                    all_pair_results[locs[i], locs[j]] = 1.0
                    all_pair_results[locs[j], locs[i]] = 1.0


        for key in all_pair_results:
            if all_pair_results[key] != 0.0:
                pair_results[key] = all_pair_results[key]

        return pair_results