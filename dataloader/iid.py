import numpy as np

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