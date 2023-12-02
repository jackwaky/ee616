import torch
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
from tqdm import tqdm


def distribute_iid_data(dataset, num_clients, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    print(f'Number of samples per client: {samples_per_client}')

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    distributed_data = {}
    for i in tqdm(range(num_clients)):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_samples = indices[start_idx:end_idx]

        distributed_data[i] = set(client_samples)

    return distributed_data

def distrubte_non_iid_domain(dataset, num_clients, domain_num_per_client, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    print(f'Number of samples per client : {samples_per_client}')

    domain_labels = dataset.tensors[2]
    unique_domains = np.unique(domain_labels)

    distributed_data = {}
    selected_samples = set()  # Keep track of selected samples

    for i in tqdm(range(num_clients)):
        num_client_samples = 0
        # If the number of selected samples is less than required, sample with replacement
        while (num_client_samples < samples_per_client):
            # Randomly select a fixed number of unique domains for each client
            client_domains = np.random.choice(unique_domains, size=domain_num_per_client, replace=False)
            client_samples = np.where(np.isin(domain_labels, client_domains))[0]

            # Exclude already selected samples
            client_samples = np.setdiff1d(client_samples, list(selected_samples), assume_unique=True)

            num_client_samples = len(client_samples)

        np.random.shuffle(client_samples)
        distributed_data[i] = set(client_samples[:samples_per_client])

        # Update selected samples
        selected_samples.update(distributed_data[i])

    return distributed_data

def distribute_non_iid_class(dataset, num_clients, label_num_per_client, ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    print(f'Number of samples per client : {samples_per_client}')

    class_labels = dataset.tensors[1]
    unique_classes = np.unique(class_labels)

    distributed_data = {}
    selected_samples = set()

    for i in tqdm(range(num_clients)):
        num_client_samples = 0
        # If the number of selected samples is less than required, sample with replacement
        while(num_client_samples < samples_per_client):
            # Randomly select a fixed number of unique classes for each client
            client_classes = np.random.choice(unique_classes, size=label_num_per_client, replace=False)
            client_samples = np.where(np.isin(class_labels, client_classes))[0]

            # Exclude already selected samples
            client_samples = np.setdiff1d(client_samples, list(selected_samples), assume_unique=True)

            num_client_samples = len(client_samples)

        np.random.shuffle(client_samples)
        distributed_data[i] = set(client_samples[:samples_per_client])

        # Update selected samples
        selected_samples.update(distributed_data[i])

    return distributed_data


def distribute_non_iid_both(dataset, num_clients, label_num_per_client_class, label_num_per_client_domain,
                            ratio_samples_per_client):
    num_samples = len(dataset)
    samples_per_client = int(ratio_samples_per_client * (num_samples // num_clients))

    print(f'Number of samples per client : {samples_per_client}')

    class_labels = dataset.tensors[1]
    domain_labels = dataset.tensors[2]

    distributed_data = {}
    selected_samples = set()

    for i in tqdm(range(num_clients)):
        num_client_samples = 0
        # If the number of selected samples is less than required, sample with replacement
        while (num_client_samples < samples_per_client):
            # Randomly select a fixed number of unique class labels for each client
            client_classes = np.random.choice(np.unique(class_labels), size=label_num_per_client_class, replace=False)
            # Randomly select a fixed number of unique domain labels for each client
            client_domains = np.random.choice(np.unique(domain_labels), size=label_num_per_client_domain, replace=False)

            # Select samples that match both selected class and domain labels
            client_samples = np.where(np.isin(class_labels, client_classes) & np.isin(domain_labels, client_domains))[0]

            # Exclude already selected samples
            client_samples = np.setdiff1d(client_samples, list(selected_samples), assume_unique=True)

            num_client_samples = len(client_samples)

        np.random.shuffle(client_samples)
        distributed_data[i] = set(client_samples[:samples_per_client])

        # Update selected samples
        selected_samples.update(distributed_data[i])

    return distributed_data