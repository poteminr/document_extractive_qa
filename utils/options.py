import argparse


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default='dataset/train.json', type=str, help='json file_path')
    parser.add_argument("--encoder_model", default='cointegrated/rubert-tiny2', type=str, help='encoder model hf path')
    parser.add_argument("--max_instances", default=-1, type=int, help='max_instances for DocumentDataset')
    parser.add_argument("--test_size", default=0.25, type=float, help='test_size for DocumentDataset')

    return parser.parse_args()
