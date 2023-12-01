import copy

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
# Dataloader
from dataloader.dataloader import fl_get_train_valid_test_dataset

# Models
from model.get_model import get_model

# Server
from server.server import Server
from server.MOON_server import MOON_Server

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
    train_dataset, test_dataset, user_group_train, user_group_test = fl_get_train_valid_test_dataset(args,
                                                                                                    train_ratio=None
                                                                                                    )
    num_clients = args.num_user
    
    fig, axs = plt.subplots(num_clients, 2, figsize=(15, 3 * num_clients), sharex=True)

    for i in range(num_clients):
        client_samples = user_group_train[i]
        class_labels = train_dataset.tensors[1][list(client_samples)]
        domain_labels = train_dataset.tensors[2][list(client_samples)]

        axs[i, 0].hist(class_labels, bins=np.arange(11) - 0.5, rwidth=0.8, color='blue', alpha=0.7)
        axs[i, 0].set_title(f'Client {i} - Class Labels')
        axs[i, 0].set_ylabel('Count')

        axs[i, 1].hist(domain_labels, bins=np.arange(len(np.unique(train_dataset.tensors[2]))) + 1.5, rwidth=0.8, color='orange', alpha=0.7)
        axs[i, 1].set_title(f'Client {i} - Domain Labels')
        axs[i, 1].set_ylabel('Count')

    print('printed!')
    plt.xlabel('Label')
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)