import argparse
import os

import joblib
import yaml

from src.data.dataset import get_metadata, get_ratings
from src.models.svd import recommend_svd


def predict_svd(config_path, user_id):
    config = yaml.safe_load(open(config_path))
    metadata_path = config['preprocess_svd']['metadata_path']
    ratings_path = config['preprocess_svd']['ratings_path']
    model_name = config['train_and_test_svd']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    algorithm = joblib.load(os.path.join(models_folder, model_name))

    metadata = get_metadata(metadata_path)
    metadata['movieId'] = metadata['movieId'].astype(int)
    title_to_id = dict(zip(metadata.title.tolist(), metadata.movieId.tolist()))

    ratings = get_ratings(ratings_path)

    predicted_ratings = recommend_svd(algorithm, user_id, ratings, title_to_id)
    print(predicted_ratings)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--user_id', dest='user_id', required=True)
    args = args_parser.parse_args()

    predict_svd(args.config, args.user_id)
