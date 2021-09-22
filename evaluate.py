import numpy as np
import scipy
import scipy.spatial


def fx_calc_map_label(image, text, label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    sim = (np.dot(label, label.T) > 0).astype(float)
    tindex = np.arange(numcases, dtype=float) + 1
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]
        num = sim[i].sum()
        a = np.where(sim[i]==1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        res += [(sim[i] / tindex).sum() / num]

    return np.mean(res)
