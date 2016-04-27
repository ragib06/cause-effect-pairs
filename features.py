import numpy as np
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.special import psi
from scipy.stats.stats import pearsonr, chisquare, f_oneway, kruskal
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, dice, hamming, jaccard, kulsinski, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, sqeuclidean, yule
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_completeness_v_measure, homogeneity_score, mutual_info_score, normalized_mutual_info_score, v_measure_score
from collections import defaultdict
import math
from scipy.stats import entropy
from numpy.linalg import norm

def enum(**enums):
    return type('Enum', (), enums)

FeatureDataType = enum(NUMERICAL=0, CATEGORICAL=1, BINARY=2)


def get_feature_data_type(type_string):
    if type_string == "Numerical":
        return FeatureDataType.NUMERICAL
    elif type_string == "Categorical":
        return FeatureDataType.CATEGORICAL
    else:
        return FeatureDataType.BINARY


def check_nn(Atype, Btype):
    return Atype == FeatureDataType.NUMERICAL and Btype == FeatureDataType.NUMERICAL


def nn_braycurtis(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return braycurtis(x, y)
    else:
        return 0

def nn_canberra(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return canberra(x, y)
    else:
        return 0

def nn_chebyshev(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return chebyshev(x, y)
    else:
        return 0

def nn_cityblock(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return cityblock(x, y)
    else:
        return 0

def nn_correlation(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return correlation(x, y)
    else:
        return 0

def nn_cosine(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return cosine(x, y)
    else:
        return 0

def nn_euclidean(x, y, xt, yt):
    if check_nn(xt, yt):
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        return euclidean(x, y)
    else:
        return 0

def anova(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    grouped_values = grouped.values()
    if len(grouped_values) < 2:
        return (0, 0, 0, 0)
    f_oneway_res = list(f_oneway(*grouped_values))
    try:
        kruskal_res = list(kruskal(*grouped_values))
    except ValueError:  # when all numbers are identical
        kruskal_res = [0, 0]
    return (f_oneway_res + kruskal_res)


def anova_f_oneway_stat(x, y, xt, yt):
    if check_nn(xt, yt):
        afos = anova(x, y)
        return afos[0]
    else:
        return 0


def anova_kruskal_stat(x, y):
    return anova(x, y)[2]


'''' def nn_JSD(x, y, xt, yt):
   if check_nn(xt,yt):
     _P = P / norm(P, ord=1)
     _Q = Q / norm(Q, ord=1)
     _M = 0.5 * (_P + _Q)
     return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
   else:
     return 0 '''


NN_FEATURES = {
    nn_braycurtis,
    nn_canberra,
    nn_chebyshev,
    nn_cityblock,
    nn_correlation,
    nn_cosine,
    nn_euclidean,
}

def check_bb(Atype, Btype):
    return Atype == FeatureDataType.BINARY and Btype == FeatureDataType.BINARY

def bb_dice(x, y, xt, yt):
    if check_bb(xt, yt):
        return dice(x, y)
    else:
        return 0

def bb_hamming(x, y, xt, yt):
    if check_bb(xt, yt):
        return hamming(x, y)
    else:
        return 0

def bb_jaccard(x, y, xt, yt):
    if check_bb(xt, yt):
        return jaccard(x, y)
    else:
        return 0

def bb_kulsinski(x, y, xt, yt):
    if check_bb(xt, yt):
        return kulsinski(x, y)
    else:
        return 0

def bb_matching(x, y, xt, yt):
    if check_bb(xt, yt):
        return matching(x, y)
    else:
        return 0

def bb_rogerstanimoto(x, y, xt, yt):
    if check_bb(xt, yt):
        return rogerstanimoto(x, y)
    else:
        return 0

def bb_russellrao(x, y, xt, yt):
    if check_bb(xt, yt):
        return russellrao(x, y)
    else:
        return 0

def bb_sokalmichener(x, y, xt, yt):
    if check_bb(xt, yt):
        return sokalmichener(x, y)
    else:
        return 0

def bb_sokalsneath(x, y, xt, yt):
    if check_bb(xt, yt):
        return sokalsneath(x, y)
    else:
        return 0

def bb_sqeuclidean(x, y, xt, yt):
    if check_bb(xt, yt):
        return sqeuclidean(x, y)
    else:
        return 0

def bb_yule(x, y, xt, yt):
    if check_bb(xt, yt):
        return yule(x, y)
    else:
        return 0

BB_FEATURES = {
    bb_dice,
    bb_hamming,
    bb_jaccard,
    bb_kulsinski,
    bb_matching,
    bb_rogerstanimoto,
    bb_russellrao,
    bb_sokalmichener,
    bb_sokalsneath,
    bb_sqeuclidean,
    bb_yule
}


def check_cc(Atype, Btype):
    return Atype == FeatureDataType.CATEGORICAL and Btype == FeatureDataType.CATEGORICAL

def check_cn(Atype, Btype):
    return Atype == FeatureDataType.CATEGORICAL and Btype == FeatureDataType.NUMERICAL

def check_nc(Atype, Btype):
    return Atype == FeatureDataType.NUMERICAL and Btype == FeatureDataType.CATEGORICAL

def descretize(a, numBins=10):
    chunk = int(math.ceil(max(a)/10))
    return np.digitize(a, range(0, int(math.ceil(max(a))), chunk))

def cc_adjusted_mutual_info_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return adjusted_mutual_info_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return adjusted_mutual_info_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return adjusted_mutual_info_score(x, y)
    else:
        return 0

def cc_adjusted_rand_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return adjusted_rand_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return adjusted_rand_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return adjusted_rand_score(x, y)
    else:
        return 0

def cc_completeness_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return completeness_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return completeness_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return completeness_score(x, y)
    else:
        return 0

def cc_homogeneity_completeness_v_measure(x, y, xt, yt):
    if check_cc(xt, yt):
        return homogeneity_completeness_v_measure(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return homogeneity_completeness_v_measure(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return homogeneity_completeness_v_measure(x, y)
    else:
        return 0

def cc_homogeneity_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return homogeneity_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return homogeneity_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return homogeneity_score(x, y)
    else:
        return 0

def cc_mutual_info_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return mutual_info_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return mutual_info_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return mutual_info_score(x, y)
    else:
        return 0

def cc_normalized_mutual_info_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return normalized_mutual_info_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return normalized_mutual_info_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return normalized_mutual_info_score(x, y)
    else:
        return 0

def cc_v_measure_score(x, y, xt, yt):
    if check_cc(xt, yt):
        return v_measure_score(x, y)
    elif check_cn(xt, yt):
        y = descretize(y)
        return v_measure_score(x, y)
    elif check_nc(xt, yt):
        x = descretize(x)
        return v_measure_score(x, y)
    else:
        return 0


CC_FEATURES = {
    cc_adjusted_mutual_info_score,
    cc_adjusted_rand_score,
    cc_completeness_score,
    # cc_homogeneity_completeness_v_measure,
    cc_homogeneity_score,
    cc_mutual_info_score,
    cc_normalized_mutual_info_score,
    cc_v_measure_score
}


class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.transform(X[column_names])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

def count_unique(x):
    return len(set(x))

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    
    hx = 0.0
    for i in range(len(x)-1):
        delta = x[i+1] - x[i]
        if delta != 0:
            hx += np.log(np.abs(delta))
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1)

    return hx


def chi_square(x, y):
    cs = chisquare(x - min(x) + 1, y - min(y) + 1)
    # print cs
    return cs


def chi_square_stat(x, y):
    return chi_square(x, y)[0]


def normalized_mutual_information(x, y):
    return metrics.mutual_info_score(x, y)


def entropy_difference(x, y):
    return normalized_entropy(x) - normalized_entropy(y)


def correlation(x, y):
    return pearsonr(x, y)[0]


def correlation_magnitude(x, y):
    return abs(correlation(x, y))


class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T


class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
