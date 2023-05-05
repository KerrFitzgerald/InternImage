import torch
from config import get_config
from models import build_model
import argparse

def create_model_from_config(cfg_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    args = parser.parse_args([f'--cfg={cfg_file}'])

    config = get_config(args)
    model = build_model(config)
    model.cuda()
    return model