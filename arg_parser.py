"""
arg_parser.py
Created: 2023.09.23
"""
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="FLdetect")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset to use to train")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model structure")
    parser.add_argument("--gpu", type=int, default=7, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='Set specific gpu to train')
    parser.add_argument("--method", type=str, default="base",
                        help="Model structure")

    # For FL data distribution
    parser.add_argument("--data_distribution", type=str, default="iid",
                        help="Choose data distribution: iid, non_iid_class, non_iid_domain, non_iid")
    parser.add_argument("--num_user", type=int, default=100,
                        help="Number of artificial clients in FL training")
    parser.add_argument("--ratio_samples_per_client", type=float, default=0.5,
                        help="Ratio of num samples per client: ratio*(num_samples/num_clients)")
    parser.add_argument("--num_classes_per_client", type=int, default=2,
                        help="Number of classes each client will have in non-iid setting")
    parser.add_argument("--num_domains_per_client", type=int, default=2,
                        help="Number of domains each client will have in non-iid setting")
    
    # For FL training
    parser.add_argument("--federated_round", type=int, default=100,
                        help="Number of global rounds in FL training")
    parser.add_argument("--local_epoch", type=int, default=5,
                        help="Number of local epochs in FL training")
    parser.add_argument("--percentage_selected_client", type=float, default=0.1,
                        help="Percentage of clients to be selected every round")

    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--bs", type=int, default=64,
                        help="Batch size")

    # for MOON
    parser.add_argument("--mu", type=float, default=1,
                        help="weight for contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="temperature for contrastive loss")

    # Client Selection method
    parser.add_argument("--selection", type=str, default="random",
                        help="Client selection method in server")


    args = parser.parse_args()
    return args

