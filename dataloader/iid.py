import numpy as np
from collections import defaultdict

def iid(dataset, num_users):
    """
    Sample iid client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index (key=user_index, value=corresponding data)
    """
    if dataset == None:
        return None

    num_items = int(len(dataset) / num_users)
    dict_users, all_indices = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        # for each user, randomly select num_items number of samples from all indices, without replacement
        dict_users[i] = set(np.random.choice(all_indices, num_items, replace=False))

        # update all_indices to exclude the previous indices assigned for the previous user
        all_indices = list(set(all_indices) - dict_users[i])

    return dict_users  # dictionary of user_idx:int, sample_indices: set

def non_iid(args, dataset, num_users, num_classes_per_client, classes_to_clients):

    """
    Sample non-iid client data from dataset
    :param dataset:
    :param num_users:
    :param num_classes_per_client:
    :return: dict of image index (key=user_index, value=corresponding data)
    """
    num_classes = args.configs.MNISTOpt["num_classes"]

    labels = np.array([data[1] for data in dataset])
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    dict_users = {i : set() for i in range(num_users)}

    if classes_to_clients == None:
        classes_to_clients = allocate_clients_to_classes(num_users, num_classes_per_client, num_classes)
    num_items_per_class = {i : int(len(class_indices[i]) / len(classes_to_clients[i])) for i in range(num_classes)}
    clients_to_classes = defaultdict(list)

    for cls, clients in classes_to_clients.items():
        for client in clients:
            clients_to_classes[client].append(cls)

    print(clients_to_classes)
    if dataset == None:
        return None

    for i in range(num_users):
        for cls in clients_to_classes[i]:
            dict_users[i] |= set(np.random.choice(class_indices[cls], num_items_per_class[cls], replace=False))
            # update all_indices to exclude the previous indices assigned for the previous user
            class_indices[cls] = list(set(class_indices[cls]) - dict_users[i])

    # for i in range(num_users):
    #     print(f"# of datas in client {i} : {len(dict_users[i])}")

    return dict_users, classes_to_clients  # dictionary of user_idx:int, sample_indices: set


def allocate_clients_to_classes(num_users, num_classes_per_client, num_classes):
    """
    Allocate clients to classes based on specified parameters
    :param num_users: Total number of clients
    :param num_classes_per_client: Maximum number of classes per client
    :param num_classes: Total number of classes
    :return: Dictionary with keys as classes and values as lists of allocated clients
    """
    clients_per_class = defaultdict(list)

    # Assign classes to clients
    for user in range(num_users):
        allocated_classes = np.random.choice(range(num_classes), num_classes_per_client, replace=False)

        for cls in allocated_classes:
            clients_per_class[cls].append(user)

    return clients_per_class