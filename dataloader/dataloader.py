import numpy as np
import random
from torchvision import datasets, transforms
from .distribute import iid, generate_label_skew, generate_feature_skew
from .Cifar10Cdistribute import *
from .Cifar10Cdataset import *
from .MnistCdistrubute import *
from .MnistCdataset import *


def fl_get_train_valid_test_dataset(args, train_ratio):

    if args.dataset == 'mnist':
        args.configs = args.configs.MNISTOpt
        apply_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(args.configs['file_path'], train=True, download=True,
                                       transform=apply_transform)
        valid_dataset = None
        test_dataset = datasets.MNIST(args.configs['file_path'], train=False, download=True,
                                       transform=apply_transform)

    elif args.dataset == 'cifar10':
        args.configs = args.configs.CIFAR10Opt
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
        train_dataset = datasets.CIFAR10(args.configs['file_path'], train=True, download=True,
                                         transform=transform_train)
        valid_dataset = None
        test_dataset = datasets.CIFAR10(args.configs['file_path'], train=False, download=True,
                                        transform=transform_test)
    
    elif args.dataset == 'cifar10c':
        args.configs = args.configs.CIFAR10COpt
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10C(file_path=args.configs['file_path'], train=True,
                                         transform=transform_train)
        valid_dataset = None
        test_dataset = CIFAR10C_Dataset(file_path=args.configs['file_path'], train=False, domain='test',
                                        transform=transform_test)

    elif args.dataset == 'mnistc':
        args.configs = args.configs.MNISTCOpt
        apply_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNISTC(args.configs['file_path'], train=True, transform=apply_transform)
        valid_dataset = None
        test_dataset = datasets.MNISTC(args.configs['file_path'], train=False, transform=apply_transform)
        
    #Code for distribute dataset.
    if args.data_distribution == 'iid':
        if args.dataset == 'cifar10c' or args.dataset == 'mnistc':
            user_group_train = distribute_iid_data(train_dataset, args.num_user, args.ratio_samples_per_client)
            # NOTE_Test data distribution is strictly iid.
            user_group_test = distribute_iid_data(test_dataset, args.num_user, args.ratio_samples_per_client)
        else:
            user_group_train = iid(train_dataset, args.num_user)
            # user_group_valid = iid(valid_dataset, args.num_user)
            user_group_test = iid(test_dataset, args.num_user)
            
    elif args.data_distribution == 'non_iid_class':
        user_group_train = distribute_non_iid_class(train_dataset, args.num_user, args.num_classes_per_client, args.ratio_samples_per_client)
        # NOTE_Test data distribution is strictly iid.
        user_group_test = distribute_iid_data(test_dataset, args.num_user, args.ratio_samples_per_client)
    
    elif args.data_distribution == 'non_iid_domain':
        user_group_train = distribute_non_iid_domain(train_dataset, args.num_user, args.num_domains_per_client, args.ratio_samples_per_client)
        # NOTE_Test data distribution is strictly iid.
        user_group_test = distribute_iid_data(test_dataset, args.num_user, args.ratio_samples_per_client)

    elif args.data_distribution == 'non_iid':
        if args.dataset == 'cifar10c' or args.dataset == 'mnistc':
            user_group_train = distribute_non_iid_both(train_dataset, args.num_user, args.num_classes_per_client, args.num_domains_per_client, args.ratio_samples_per_client)
            # NOTE_Test data distribution is strictly iid.
            user_group_test = distribute_iid_data(test_dataset, args.num_user, args.ratio_samples_per_client)
        else:
            user_group_train, classes_to_clients = generate_label_skew(args, train_dataset, args.num_user, args.num_classes_per_client, None)
            # user_group_valid = non_iid(args, valid_dataset, args.num_user, args.num_classes_per_client)
            user_group_test, _ = generate_label_skew(args, test_dataset, args.num_user, args.num_classes_per_client, classes_to_clients)

            train_dataset, user_group_train, augmentations = generate_feature_skew(args, train_dataset, user_group_train, None)
            test_dataset, user_group_test, _ = generate_feature_skew(args, test_dataset, user_group_test, augmentations)


    print(f'# of train dataset : {len(train_dataset)}, # of test dataset : {len(test_dataset)}')


    return train_dataset, test_dataset, user_group_train, user_group_test