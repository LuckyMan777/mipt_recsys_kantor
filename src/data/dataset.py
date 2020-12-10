# -*- coding: utf-8 -*-

import pandas as pd


def get_metadata(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def get_links(path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_ratings(path) -> pd.DataFrame:
    return pd.read_csv(path)
