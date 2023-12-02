import copy
import time
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

import random

class MOON_Client():
    def __init__(self, args, train_dataset, test_dataset, user_datapoint_indices_train, user_datapoint_indices_test, user_idx):

        self.args = args
        self.user_idx = user_idx

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        self.train_dataloader = DataLoader(
            DatasetSplit(train_dataset, user_datapoint_indices_train), batch_size=args.bs,
            shuffle=True, drop_last=True)

        self.test_dataloader = DataLoader(test_dataset, 
            batch_size=args.bs,
            shuffle=False, drop_last=True)


    def local_train(self, model, previous_model):
        global_model = copy.deepcopy(model)

        model.to(self.args.device)
        model.train()

        previous_model.to(self.args.device)
        previous_model.eval()

        global_model.to(self.args.device)
        global_model.eval()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        cos = torch.nn.CosineSimilarity(dim=-1).to(self.args.device)

        # global_encoder = nn.Sequential(*list(global_model.children())[:-1])
        # previous_encoder = nn.Sequential(*list(previous_model.children())[:-1])
        # local_encoder = nn.Sequential(*list(model.children())[:-1])

        epoch_loss = []
        for local_epoch in range(self.args.local_epoch):
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

                # MOON
                _, feature_local, out = model(inputs)
                _, feature_pos, _ = global_model(inputs)


                posi = cos(feature_local, feature_pos)
                logits = posi.reshape(-1, 1)

                _, feature_prev, _ = previous_model(inputs)

                nega = cos(feature_local, feature_prev)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= 0.5 # temperature in MOON
                targets = torch.zeros(inputs.size(0)).to(self.args.device).long()

                loss2 = 1 * self.criterion(logits, targets)

                loss1 = self.criterion(out, labels)

                loss = loss1 + loss2

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
        input_data, label, _ = self.all_dataset[self.curr_user_indices[item]]  # only items related to the current user (not all clients)
        return input_data.clone().detach(), label.clone().detach()