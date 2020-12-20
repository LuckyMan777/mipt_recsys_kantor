import argparse
import os

import joblib
import pandas as pd
import yaml

from src.data.dataset import get_metadata
from src.models.simple_model import get_recommendations


def predict_simple_model(config_path, title):
    config = yaml.safe_load(open(config_path))
    metadata_path = config['data_load']['metadata_csv']
    metadata = get_metadata(metadata_path)
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    model_name = config['train_simple_model']['model']['model_name']
    models_folder = config['base']['model']['models_folder']
    cosine_sim = joblib.load(os.path.join(models_folder, model_name))

    print(get_recommendations(title, metadata, indices, cosine_sim))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--title', dest='title', required=True)
    args = args_parser.parse_args()

    predict_simple_model(args.config, args.title)
