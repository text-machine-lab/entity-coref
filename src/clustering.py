import sys
import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import linkage, inconsistent, fcluster
import numpy as np

def clustering(pair_results, binarize=False):
    def distance(e1, e2):
        e1 = tuple(e1.astype(int))
        e2 = tuple(e2.astype(int))
        if e1 == e2:
            return 1.0  # This is the minumum distance
        if (e1, e2) in pair_results:
            similarity = max(pair_results[(e1, e2)], 1e-3)
            # dist = 1 - pair_results[(e1, e2)] #+ 1e-4
            dist = min(1.0 / (similarity), 10.0)
            # dist = (10 * (1 - pair_results[(e1, e2)])) ** 2
        else:
            # dist = 0.9
            dist = 10.0
        if binarize:
            dist = np.round(dist)

        return dist

    # distance has no direction
    if sys.version_info[0] == 3:
        pairs = list(pair_results.keys())
    else:
        pairs = pair_results.keys()
    for key in pairs:
        pair_results[(key[1], key[0])] = pair_results[key]

    x = [key[0] for key in pair_results]
    x = list(set(x))
    x.sort()
    x = np.array(x)

    clusters, Z = fclusterdata(x, 1.6, criterion='distance', metric=distance, depth=2, method='single')
    return x, clusters, Z


def fclusterdata(X, t, criterion='inconsistent',
                     metric='euclidean', depth=2, method='single', R=None):
    """
    This is adapted from scipy fclusterdata.
    https://github.com/scipy/scipy/blob/v1.0.0/scipy/cluster/hierarchy.py#L1809-L1878
    """
    X = np.asarray(X, order='c', dtype=np.double)

    if type(X) != np.ndarray or len(X.shape) != 2:
        print(type(X), X.shape)
        raise TypeError('The observation matrix X must be an n by m numpy '
                        'array.')

    Y = distance.pdist(X, metric=metric)
    Z = linkage(Y, method=method)
    if R is None:
        R = inconsistent(Z, d=depth)
    else:
        R = np.asarray(R, order='c')
    T = fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    return T, Z

