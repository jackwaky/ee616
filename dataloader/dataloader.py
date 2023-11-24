import numpy as np
import random
from torchvision import datasets, transforms
from .iid import iid


def fl_get_train_valid_test_dataset(args, train_ratio, valid_ratio):

    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(args.configs.MNISTOpt['file_path'], train=True, download=True,
                                       transform=apply_transform)
        valid_dataset = None
        test_dataset = datasets.MNIST(args.configs.MNISTOpt['file_path'], train=False, download=True,
                                       transform=apply_transform)


    if args.data_distribution == 'iid':
        user_group_train = iid(train_dataset, args.num_user)
        user_group_valid = iid(valid_dataset, args.num_user)
        user_group_test = iid(test_dataset, args.num_user)

    print(f'# of train dataset : {len(train_dataset)}, # of test dataset : {len(test_dataset)}')

    return train_dataset, valid_dataset, test_dataset, user_group_train, user_group_valid, user_group_test