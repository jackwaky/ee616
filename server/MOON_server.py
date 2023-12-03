import time
from tqdm import tqdm
import numpy as np
import random
import copy
import torch

from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader

from client.MOON_client import MOON_Client

import copy

from utils import get_domain_info, select_clients_uniformly, select_clients_one ,make_domain_dist

class MOON_Server():
    def __init__(self, args, model, train_dataset, test_dataset, user_group_train, user_group_test):

        self.args = args
        self.global_model = model
        self.global_weights = self.global_model.state_dict()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_group_train = user_group_train
        self.user_group_test = user_group_test

        # Setting for MOON
        self.previous_nets = {i: copy.deepcopy(model) for i in self.user_group_train.keys()}

        # key : client idx, value : domain idx
        self.domain_info = get_domain_info(train_dataset, user_group_train)
        self.domain_numbers = set()
        for values in self.domain_info.values():
            self.domain_numbers.update(values)


    def train(self):
        start_time = time.time()

        # Initialize the model
        self.global_model.to(self.args.device)

        total_num_of_clients = len(self.user_group_train)
        m = int(total_num_of_clients * self.args.percentage_selected_client)
        
        major_domain = np.random.choice(list(self.domain_numbers),size=len(self.domain_numbers),replace=False)

        for epoch in range(self.args.federated_round):
            print(f'Federated Learning round {epoch + 1}/{self.args.federated_round}')

            # Client Selection (Random)
            if self.args.selection == 'random':
                selected_user_indices = random.sample(self.user_group_train.keys(), m)
            # Uniformly select the clients in domain
            elif self.args.selection == 'uniform':
                selected_user_indices = select_clients_uniformly(self.domain_info, m)
            # ratio one group & 1-ratio others -- group A in all round // random others
            elif self.args.selection == 'one_fix':
                selected_user_indices = select_clients_one(domain_info=self.domain_info, num_selected_client=m, domain_numbers=self.domain_numbers, epoch=epoch, major_domain=major_domain, rpd=self.args.federated_round, major_ratio=self.args.major_domain_ratio)
            elif self.args.selection == 'one_rand':
                selected_user_indices = select_clients_one(domain_info=self.domain_info, num_selected_client=m, domain_numbers=self.domain_numbers, epoch=epoch, major_domain=None, rpd=1, major_ratio=self.args.major_domain_ratio)
            elif self.args.selection == 'one_seq': # one_round, round=1 
                selected_user_indices = select_clients_one(domain_info=self.domain_info, num_selected_client=m, domain_numbers=self.domain_numbers, epoch=epoch, major_domain=major_domain, rpd=1, major_ratio=self.args.major_domain_ratio)
            elif self.args.selection == 'one_round_rand':
                selected_user_indices = select_clients_one(domain_info=self.domain_info, num_selected_client=m, domain_numbers=self.domain_numbers, epoch=epoch, major_domain=None, rpd=self.args.rounds_per_domain, major_ratio=self.args.major_domain_ratio)
            elif self.args.selection == 'one_round_seq':
                selected_user_indices = select_clients_one(domain_info=self.domain_info, num_selected_client=m, domain_numbers=self.domain_numbers, epoch=epoch, major_domain=major_domain, rpd=self.args.rounds_per_domain, major_ratio=self.args.major_domain_ratio)

            print(f'In round {epoch + 1}, # of selected clients : {len(selected_user_indices)}, selected clients are {selected_user_indices}')
            
            selected_domains = make_domain_dist(self.domain_info,selected_user_indices)
            
            print(f'Selected domain distribution: {selected_domains}')
            local_weight_list, local_loss_list = [], []

            for cur_client_idx in selected_user_indices:
                local_client = MOON_Client(args = self.args,
                                           train_dataset = self.train_dataset,
                                           test_dataset = self.test_dataset,
                                           user_datapoint_indices_train = self.user_group_train[cur_client_idx],
                                           user_datapoint_indices_test = self.user_group_test[cur_client_idx],
                                           user_idx = cur_client_idx)

                local_weight, local_loss = local_client.local_train(model = copy.deepcopy(self.global_model),
                                                       previous_model = copy.deepcopy(self.previous_nets[cur_client_idx]))

                local_weight_list.append(copy.deepcopy(local_weight))
                local_loss_list.append(local_loss)

                # Update previous local model
                self.previous_nets[cur_client_idx].load_state_dict(local_weight)

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
            for batch_idx, (inputs, labels, _) in enumerate(testloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

                _, _, outputs = self.global_model(inputs)

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



