# -*- coding: utf-8 -*-

import pandas as pd


def get_metadata() -> pd.DataFrame:
    return pd.read_csv('../data/external/movies_metadata.csv', low_memory=False)


def get_links() -> pd.DataFrame:
    return pd.read_csv('../data/external/links.csv')


def get_ratings() -> pd.DataFrame:
    return pd.read_csv('../data/external/ratings_small.csv')
