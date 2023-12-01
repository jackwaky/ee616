import torch
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np

def distribute_iid_data(dataset, num_clients, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    distributed_data = {}
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_samples = indices[start_idx:end_idx]

        distributed_data[i] = set(client_samples)

    return distributed_data

def distribute_non_iid_domain(dataset, num_clients, domain_num_per_client, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    domain_labels = dataset.tensors[2]
    unique_domains = np.unique(domain_labels)

    distributed_data = {}
    for i in range(num_clients):
        # Randomly select a fixed number of unique domains for each client
        client_domains = np.random.choice(unique_domains, size=domain_num_per_client, replace=False)
        client_samples = np.where(np.isin(domain_labels, client_domains))[0]

        # If the number of selected samples is less than required, sample with replacement
        if len(client_samples) < samples_per_client:
            remaining_samples = samples_per_client - len(client_samples)
            additional_samples = np.random.choice(num_samples, size=remaining_samples, replace=True)
            client_samples = np.concatenate((client_samples, additional_samples))

        np.random.shuffle(client_samples)
        distributed_data[i] = set(client_samples)

    return distributed_data

def distribute_non_iid_class(dataset, num_clients, label_num_per_client, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    class_labels = dataset.tensors[1]
    unique_classes = np.unique(class_labels)

    distributed_data = {}
    for i in range(num_clients):
        # Randomly select a fixed number of unique classes for each client
        client_classes = np.random.choice(unique_classes, size=label_num_per_client, replace=False)
        client_samples = np.where(np.isin(class_labels, client_classes))[0]

        # If the number of selected samples is less than required, sample with replacement
        if len(client_samples) < samples_per_client:
            remaining_samples = samples_per_client - len(client_samples)
            additional_samples = np.random.choice(num_samples, size=remaining_samples, replace=True)
            client_samples = np.concatenate((client_samples, additional_samples))

        np.random.shuffle(client_samples)
        distributed_data[i] = set(client_samples)

    return distributed_data

def distribute_non_iid_both(dataset, num_clients, label_num_per_client_class, label_num_per_client_domain, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))
    
    class_labels = dataset.tensors[1]
    domain_labels = dataset.tensors[2]

    distributed_data = {}
    for i in range(num_clients):
        # Randomly select a fixed number of unique class labels for each client
        client_classes = np.random.choice(np.unique(class_labels), size=label_num_per_client_class, replace=False)
        # Randomly select a fixed number of unique domain labels for each client
        client_domains = np.random.choice(np.unique(domain_labels), size=label_num_per_client_domain, replace=False)

        # Select samples that match both selected class and domain labels
        client_samples = np.where(np.isin(class_labels, client_classes) & np.isin(domain_labels, client_domains))[0]

        # If the number of selected samples is less than required, sample with replacement
        if len(client_samples) < samples_per_client:
            remaining_samples = samples_per_client - len(client_samples)
            additional_samples = np.random.choice(num_samples, size=remaining_samples, replace=True)
            client_samples = np.concatenate((client_samples, additional_samples))

        np.random.shuffle(client_samples)
        distributed_data[i] = set(client_samples)

    return distributed_data
