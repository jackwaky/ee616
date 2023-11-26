import numpy as np
import random
from torchvision import datasets, transforms
from .distribute import iid, generate_label_skew, generate_feature_skew


def fl_get_train_valid_test_dataset(args, train_ratio):

    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(args.configs.MNISTOpt['file_path'], train=True, download=True,
                                       transform=apply_transform)
        valid_dataset = None
        test_dataset = datasets.MNIST(args.configs.MNISTOpt['file_path'], train=False, download=True,
                                       transform=apply_transform)

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(args.configs.CIFAR10Opt['file_path'], train=True, download=True,
                                         transform=transform_train)
        valid_dataset = None
        test_dataset = datasets.CIFAR10(args.configs.CIFAR10Opt['file_path'], train=False, download=True,
                                        transform=transform_test)

    if args.data_distribution == 'iid':
        user_group_train = iid(train_dataset, args.num_user)
        # user_group_valid = iid(valid_dataset, args.num_user)
        user_group_test = iid(test_dataset, args.num_user)

    elif args.data_distribution == 'non_iid':
        user_group_train, classes_to_clients = generate_label_skew(args, train_dataset, args.num_user, args.num_classes_per_client, None)
        # user_group_valid = non_iid(args, valid_dataset, args.num_user, args.num_classes_per_client)
        user_group_test, _ = generate_label_skew(args, test_dataset, args.num_user, args.num_classes_per_client, classes_to_clients)

        train_dataset, user_group_train, augmentations = generate_feature_skew(args, train_dataset, user_group_train, None)
        test_dataset, user_group_test, _ = generate_feature_skew(args, test_dataset, user_group_test, augmentations)


    print(f'# of train dataset : {len(train_dataset)}, # of test dataset : {len(test_dataset)}')


    return train_dataset, test_dataset, user_group_train, user_group_test