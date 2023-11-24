import copy

import torch
import numpy as np
import random
import os

# Dataloader
from dataloader.dataloader import fl_get_train_valid_test_dataset

# Models
from model.get_model import get_model

# Server
from server.server import Server

import configs
from arg_parser import parse_arguments

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    set_seed(args.seed)
    args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
    args.configs = configs
    print(f"START!\nargs:\n{args}\n")

    # 1. Load and Split the dataset into train / valid / test with ratio, and get user_group (for FL)
    train_dataset, valid_dataset, test_dataset, \
        user_group_train, user_group_valid, user_group_test = fl_get_train_valid_test_dataset(args,
                                                                                              train_ratio=None,
                                                                                              valid_ratio=None
                                                                                              )

    # # 2. Define the model
    # model = get_model(args.model, args)
    model = get_model(args.model, args)


    server = Server(args, model, train_dataset, test_dataset, user_group_train, user_group_test)
    server.train()
    server.test()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)