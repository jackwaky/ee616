import time
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

import random

class Client():
    def __init__(self, args, train_dataset, test_dataset, user_datapoint_indices_train, user_datapoint_indices_test, user_idx):

        self.args = args
        self.user_idx = user_idx

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        self.train_dataloader = DataLoader(
            DatasetSplit(train_dataset, user_datapoint_indices_train), batch_size=args.bs,
            shuffle=True, drop_last=True)

        self.test_dataloader = DataLoader(
            DatasetSplit(test_dataset, user_datapoint_indices_test),
            batch_size=args.bs,
            shuffle=False, drop_last=True)


    def local_train(self, model, global_round):
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        epoch_loss = []
        for local_epoch in range(self.args.local_epoch):
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

class DatasetSplit(Dataset):
    def __init__(self, all_dataset, curr_user_indices):
        self.all_dataset = all_dataset  # data of all clients
        self.curr_user_indices = [int(i) for i in curr_user_indices]  # indices corresponding to this user only

    def __len__(self):
        return len(self.curr_user_indices)

    def __getitem__(self, item):
        input_data, label = self.all_dataset[self.curr_user_indices[item]]  # only items related to the current user (not all clients)
        return torch.tensor(input_data), torch.tensor(label)