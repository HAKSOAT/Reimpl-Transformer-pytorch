import json
import random
from argparse import ArgumentParser
from datetime import datetime

import torch
import numpy as np

from dictionaries import IndexDictionary
from models import build_model
from utils.log import get_logger

parser = ArgumentParser(description="Train Transformer")
parser.add_argument("--config", type=str, default=None)

parser.add_argument("--data_dir", type=str, default="data/example/processed")
parser.add_argument("--dave_config", type=str, default=None)
parser.add_argument("--save_checkpoint", type=str, default=None)
parser.add_argument("--save_log", type=str, default=None)

parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available else "cpu")

parser.add_argument("--dataset_limit", type=int, default=None)
parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--save_every", type=int, default=1)

parser.add_argument("--vocabulary_size", type=int, default=None)
# Q: What does this positional encoding flag do?
parser.add_argument("--positional_encoding", action="store_true")

parser.add_argument("--d_model", type=int, default=128)
# Q: What do layers count and heads count mean?
parser.add_argument("--layers_count", type=int, default=1)
parser.add_argument("--heads_count", type=int, default=2)
# Q: What does d_ff mean?
parser.add_argument("--d_ff", type=int, default=128)
parser.add_argument("--dropout_prob", type=float, default=0.1)

# Q: What does label smoothing mean?
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Noam", "Adam"])
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--clip_grads", action="store_true")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)

def run_trainer(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name_format = (
        "d_model={d_model}"
        "layers_count={layers_count}-"
        "heads_count={heads_count}-"
        "pe={positional_encoding}-"
        "optimizer={optimizer}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**config)

    logger = get_logger(run_name, log_path=config["save_log"])
    logger.info(f"Run name: {run_name}")
    logger.info(config)

    logger.info("Constructing Dictionaries")
    source_dictionary = IndexDictionary.load(config["data_dir"], mode="source",
                                             vocabulary_size=config["vocabulary_size"])
    target_dictionary = IndexDictionary.load(config["data_dir"], mode="target",
                                             vocabulary_size=config["vocabulary_size"])
    logger.info(f"Source dictionary vocabulary: {source_dictionary.vocabulary_size} tokens")
    logger.info(f"Target dictionary vocabulary: {target_dictionary.vocabulary_size} tokens")

    logger.info("Building model...")
    model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)

    run_trainer(config)