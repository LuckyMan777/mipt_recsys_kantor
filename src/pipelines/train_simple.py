import argparse
import os

import joblib
import yaml
from sklearn.metrics.pairwise import linear_kernel

from src.data.dataset import get_metadata
from src.features.build_features import tfidf_transform


def train_simple_model(config_path):
    config = yaml.safe_load(open(config_path))
    metadata_path = config['data_load']['metadata_csv']
    metadata = get_metadata(metadata_path)
    metadata['overview'] = metadata['overview'].fillna('')

    max_features = config['train_simple_model']['tfidf']['max_features']
    tfidf_matrix = tfidf_transform(metadata, max_features)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    model_name = config['train_simple_model']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    joblib.dump(
        cosine_sim,
        os.path.join(models_folder, model_name)
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_simple_model(args.config)
