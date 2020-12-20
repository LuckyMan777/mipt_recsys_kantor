import argparse
import os

import joblib
import yaml
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from src.data.dataset import get_ratings
from src.models.svd import train_svd
from src.evaluate.evaluate import evaluate_svd


def train_and_test_svd(config_path):
    config = yaml.safe_load(open(config_path))
    ratings_path = config['preprocess_svd']['ratings_path']
    random_state = config['base']['random_state']
    test_size = config['train_and_test_svd']['test_size']

    ratings = get_ratings(ratings_path)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    algorithm = train_svd(trainset, random_state)

    print(evaluate_svd(algorithm, testset))

    model_name = config['train_and_test_svd']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    joblib.dump(
        algorithm,
        os.path.join(models_folder, model_name)
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_and_test_svd(args.config)
