import argparse


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default='dataset/train.json', type=str, help='json file_path')
    parser.add_argument("--test_file_path", default='dataset/test.json', type=str, help='test json file_path')
    parser.add_argument("--encoder_model", default='cointegrated/rubert-tiny2', type=str, help='encoder model hf path')
    parser.add_argument("--max_instances", default=-1, type=int, help='max_instances for DocumentDataset')
    parser.add_argument("--test_size", default=0.25, type=float, help='test_size for DocumentDataset')
    parser.add_argument("--model_pretrained_path", default=None, type=str, help='path to trained model for QA task')
    parser.add_argument("--model_type", default='huggingface', type=str, help='model type: hf or default pytorch')
    parser.add_argument("--train_label", default=-1, type=int, help='choose specific label for training')
    parser.add_argument("--seed", default=1007, type=int, help='set seed')
    return parser.parse_args()
