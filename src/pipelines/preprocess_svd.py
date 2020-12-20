import argparse

import pandas as pd
import yaml

from src.data.dataset import get_links, get_metadata, get_ratings


def preprocess_svd(config_path):
    config = yaml.safe_load(open(config_path))
    metadata_csv = config['data_load']['metadata_csv']
    links_csv = config['data_load']['links_csv']
    ratings_csv = config['data_load']['ratings_csv']
    metadata_path = config['preprocess_svd']['metadata_path']
    links_path = config['preprocess_svd']['links_path']
    ratings_path = config['preprocess_svd']['ratings_path']

    metadata = get_metadata(metadata_csv)
    metadata['overview'] = metadata['overview'].fillna('')

    links = get_links(links_csv)
    links = links.dropna()
    links = links.drop_duplicates(subset=["tmdbId"], keep='first')
    links.to_csv(links_path)

    id_to_movieId = dict(links[['tmdbId', 'movieId']].values)
    metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
    metadata = metadata.dropna(subset=['id'])
    metadata['movieId'] = metadata['id'].map(id_to_movieId).astype(int)
    metadata.to_csv(metadata_path)

    ratings = get_ratings(ratings_csv)
    correct_users = ratings.userId.unique()[ratings.groupby('userId')['rating'].count() >= 50]
    ratings = ratings[ratings.userId.isin(correct_users)]
    ratings.to_csv(ratings_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess_svd(args.config)
