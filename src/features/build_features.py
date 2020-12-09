import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def calc_weighted_rating(x, m, C):
    #     print(x)
    v = x['vote_count']
    R = x['vote_average']

    wr = v * R / (v + m) + m * C / (v + m)
    return wr


def tfidf_transform(metadata):
    tfidf = TfidfVectorizer(stop_words='english', dtype=np.float32, max_features=40000)
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    return tfidf_matrix
