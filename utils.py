import time
from tqdm import tqdm
import numpy as np
import random
import copy
import torch

from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    def __init__(self, all_dataset, curr_user_indices):
        self.all_dataset = all_dataset  # data of all clients
        self.curr_user_indices = [int(i) for i in curr_user_indices]  # indices corresponding to this user only

    def __len__(self):
        return len(self.curr_user_indices)

    def __getitem__(self, item):
        input_data, label, domain = self.all_dataset[self.curr_user_indices[item]]  # only items related to the current user (not all clients)
        return input_data.clone().detach(), label.clone().detach(), domain.clone().detach()

def get_domain_info(dataset, user_group):
    client_idx = user_group.keys()

    # key : client idx, value : domain idx
    domain_info = {}
    for cur_client in client_idx:
        cur_domain_list = []
        cur_user_dataset = DatasetSplit(dataset, user_group[cur_client])
        for data in cur_user_dataset:
            cur_domain_list.append(data[2].item())
        cur_domain_list = set(cur_domain_list)
        domain_info[cur_client] = cur_domain_list

    return domain_info

def select_clients_uniformly(domain_info, num_selected_client):
    # Create a new dictionary with only one domain per client
    single_domain_info = {client: list(domains)[0] for client, domains in domain_info.items()}

    selected_clients = []
    remaining_clients = list(single_domain_info.keys())

    while len(selected_clients) < num_selected_client and remaining_clients:
        # Randomly select a client index
        client_index = np.random.choice(remaining_clients)

        # Check if the selected client's domain does not overlap with any existing selected client
        selected_domain = single_domain_info[client_index]
        overlap = any(selected_domain == single_domain_info[c] for c in selected_clients)

        # If no overlap, add the client index to the selected list
        if not overlap:
            selected_clients.append(client_index)

        # Remove the selected client from the remaining clients
        remaining_clients.remove(client_index)

    # verify = []
    # for client in selected_clients:
    #     verify.append(single_domain_info[client])
    # print(len(set(verify)))
    #
    # verify = []
    # for client in selected_clients:
    #     verify.append(domain_info[client])
    # print(verify)

    return selected_clients
