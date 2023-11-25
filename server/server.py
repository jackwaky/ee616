import time
from tqdm import tqdm
import numpy as np
import random
import copy
import torch

from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader

from client.client import Client

class Server():
    def __init__(self, args, model, train_dataset, test_dataset, user_group_train, user_group_test):

        self.args = args
        self.global_model = model
        self.global_weights = self.global_model.state_dict()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_group_train = user_group_train
        self.user_group_test = user_group_test


    def train(self):
        start_time = time.time()

        # Initialize the model
        self.global_model.to(self.args.device)

        total_num_of_clients = len(self.user_group_train)
        m = int(total_num_of_clients * self.args.percentage_selected_client)

        for epoch in tqdm(range(self.args.federated_round)):
            print(f'Federated Learning round {epoch + 1}/{self.args.federated_round}')

            # Client Selection (Random)
            # selected_user_indices = np.random.choice(list(self.user_group_train.keys()), size=m, replace=False)
            selected_user_indices = random.sample(self.user_group_train.keys(), m)
            print(f'In round {epoch + 1}, # of selected clients : {len(selected_user_indices)}, selected clients are {selected_user_indices}')
            local_weight_list, local_loss_list = [], []

            for cur_client_idx in selected_user_indices:
                local_client = Client(args = self.args,
                                           train_dataset = self.train_dataset,
                                           test_dataset = self.test_dataset,
                                           user_datapoint_indices_train = self.user_group_train[cur_client_idx],
                                           user_datapoint_indices_test = self.user_group_test[cur_client_idx],
                                           user_idx = cur_client_idx)

                local_weight, local_loss = local_client.local_train(model = copy.deepcopy(self.global_model),
                                                       global_round = epoch)

                local_weight_list.append(copy.deepcopy(local_weight))
                local_loss_list.append(local_loss)

            self.global_weights = self.average_weights(local_weight_list)
            self.global_model.load_state_dict(self.global_weights)

            loss_avg = sum(local_loss_list) / len(local_loss_list)

            print(f'\n Avg training stats after {epoch + 1}-th global round: ')
            print(f'Training loss: {loss_avg}')

        print('\n Total run time: {0:0.4f}'.format(time.time() - start_time))

    def test(self):
        self.global_model.eval()

        loss, total, correct = 0.0, 0.0, 0.0

        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        testloader = DataLoader(self.test_dataset, batch_size=self.args.bs, shuffle=False)

        predicted_labels_list = []
        correct_labels_list = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

                outputs = self.global_model(inputs)

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                predicted_labels_list = np.concatenate((predicted_labels_list, pred_labels.cpu()), axis=0)
                correct_labels_list = np.concatenate((correct_labels_list, labels.cpu()), axis=0)

        accuracy = correct / total
        bacc = balanced_accuracy_score(correct_labels_list, predicted_labels_list)

        print(f'| --- Test Acc: {(100 * accuracy):.2f}%')
        print(f'| --- Test BAcc: {(100 * bacc):.2f}%')

        return accuracy, bacc

    def average_weights(self, local_weight_list):
        # fed average algorithm here
        w_avg = copy.deepcopy(local_weight_list[0])
        for key in w_avg.keys():
            for i in range(1, len(local_weight_list)):
                w_avg[key] += local_weight_list[i][key]
            w_avg[key] = torch.div(w_avg[key], len(local_weight_list))
        return w_avg



