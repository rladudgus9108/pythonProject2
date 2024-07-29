import numpy as np
import copy
# import matplotlib.pyplot as plt
import torch

from datasets.dataSpilt import CustomImageDataset
from datasets.FMloader import DataLoader
from options import opt
from roles.client import Client
from roles.aggregator import Cloud
from datasets.dataSpilt import get_data_loaders, get_default_data_transforms
from modules.fedprox import GenerateLocalEpochs

import matplotlib.pyplot as plt


class FmpuTrainer:
    def __init__(self, model_pu):
        # load data
        if not opt.useFedmatchDataLoader:  # default 로 usedFedmatchDatLoader : False가 되어 있음
            # create Clients and Aggregating Server
            local_dataloaders, local_sample_sizes, test_dataloader, indexlist, priorlist = get_data_loaders()
            # 윗줄에서 client별 들어가는 class와 거기서 select되는 class들 정해지는게 다 처리됨

            self.clients = [Client(_id + 1, copy.deepcopy(model_pu).cuda(), local_dataloaders[_id], test_dataloader,
                                   priorlist=priorList, indexlist=indexList)  # 여기에서 처음에 deepcopy해서 client 설정
                            for _id, priorList, indexList, in zip(list(range(opt.num_clients)), priorlist, indexlist)]
        else:
            self.loader = DataLoader(opt)
            # test_dataset = self.loader(get_test)
            # TODO: change to dataloader format
            indexlist = torch.Tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 100).cuda()
            priorlist = torch.Tensor([[0.1] * 10] * 100).cuda()
            self.load_data()
            self.loader.get_test()
            _, transforms_eval = get_default_data_transforms(opt.dataset, verbose=False)
            # test_dataset = CustomImageDataset(self.x_test, self.y_test, transforms_eval)
            test_dataset = CustomImageDataset(self.x_test.astype(np.float32) / 255, self.y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batchsize, shuffle=True)

            self.clients = [Client(_id, copy.deepcopy(model_pu).cuda(), priorlist=priorList, indexlist=indexList)
                            for _id, priorList, indexList, in zip(list(range(opt.num_clients)), priorlist, indexlist)]
            print("numclients:", opt.num_clients, "build clients:", len(self.clients))

        self.clientSelect_idxs = []

        self.cloud = Cloud(self.clients, model_pu, opt.num_classes, test_dataloader)
        self.communication_rounds = opt.communication_rounds
        self.current_round = 0

    def load_data(self):
        # for Fedmatch dataloader
        self.x_train, self.y_train, self.task_name = None, None, None
        self.x_valid, self.y_valid = self.loader.get_valid()
        self.x_test, self.y_test = self.loader.get_test()
        # self.x_test = self.loader.scale(self.x_test).transpose(0,3,1,2)
        self.x_test = self.x_test.transpose(0, 3, 1, 2)
        self.y_test = torch.argmax(torch.from_numpy(self.y_test), -1).numpy()
        self.x_valid = self.loader.scale(self.x_valid)

    def begin_train(self):

        acc_per_round = []
        percentage_of_relabeling_rate = 0.1
        print(f"Relabeling rate in batch percentage : {percentage_of_relabeling_rate * 100}%")
        for t in range(self.communication_rounds):
            self.current_round = t + 1
            self.cloud_lastmodel = self.cloud.aggregated_client_model
            self.clients_select()  # random choice로 clients select, but 1.0으로 되어 있어서 모두 다 선택됨

            if 'SL' in opt.task:  # default : opt.method 라고 되어 있었음
                print("##### Full labeled setting #####")
                self.clients_train_step_SL()
            else:
                print("##### Semi-supervised setting #####")
                self.clients_train_step_SS_suggest(percentage_of_relabeling_rate)

            self.cloud.aggregate(self.clientSelect_idxs)
            acc_per_round.append(self.cloud.validation(t))

            max_acc = max(acc_per_round)
            max_acc_round = acc_per_round.index(max_acc)
            print("Hightest Round:{}, Highest Accuracy: {:.4f} %".format(max_acc_round, max_acc))
            print("__________________________________________")

        # plt.plot(acc_per_round, label="SL batch : 1024")
        # plt.xlabel("round")
        # plt.ylabel("Acc")
        #
        # plt.legend()
        # plt.show()
        print(acc_per_round)

    def clients_select(self):
        # default
        # m = max(int(opt.clientSelect_Rate * opt.num_clients), 1)
        # self.clientSelect_idxs = np.random.choice(range(opt.num_clients), m, replace=False)

        self.clientSelect_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # default
        # self.clientSelect_idxs = [0, 1, 2, 3, 4]
        # self.clientSelect_idxs = [0, 1]
        # self.clientSelect_idxs = [0 ,1, 2, 3, 4, 5, 6, 7, 8]

    def clients_train_step_SS(self):
        if 'FedProx' in opt.method:
            print("Invalid")
            percentage = opt.percentage
            mu = opt.mu
            print(f"System heterogeneity set to {percentage}% stragglers.\n")
            print(f"Picking {len(self.clientSelect_idxs)} random clients per round.\n")
            heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients),
                                                          max_epochs=opt.local_epochs)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)

            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                if opt.use_PULoss:
                    self.clients[idx].train_fedprox_pu(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                       globalmodel=self.cloud.aggregated_client_model)
                else:
                    self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                      globalmodel=self.cloud.aggregated_client_model)
        elif 'FedAvg' in opt.method:
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                if opt.use_PULoss:  # PULoss는 positive Unlabel loss를 의미하는것, else문은 positive loss를 의미
                    # self.clients[idx].train_fedavg_pu()
                    self.clients[idx].train_fedavg_pu_mod()  # opt에 PULoss를 설정하는 부분은 없음 -> config.py에서 설정함
                else:
                    print("Invalid")
                    return
        else:
            return

    def clients_train_step_SS_suggest(self, percentage_of_relabeling_rate):
        if 'FedProx' in opt.method:
            print("Invalid")
            percentage = opt.percentage
            mu = opt.mu
            print(f"System heterogeneity set to {percentage}% stragglers.\n")
            print(f"Picking {len(self.clientSelect_idxs)} random clients per round.\n")
            heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients),
                                                          max_epochs=opt.local_epochs)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)

            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                if opt.use_PULoss:
                    self.clients[idx].train_fedprox_pu(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                       globalmodel=self.cloud.aggregated_client_model)
                else:
                    self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                      globalmodel=self.cloud.aggregated_client_model)
        elif 'FedAvg' in opt.method:
            print("Suggest SS")
            client_logit_store = []
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                client_logit_store.append(self.clients[idx].send_logit())

            global_logit = self.cloud.calculate_global_logit(client_logit_store)

            for idx in self.clientSelect_idxs:
                if opt.use_PULoss:
                    self.clients[idx].train_fedavg_pu_suggest(global_logit, percentage_of_relabeling_rate)
                    # default : self.clients[idx].train_fedavg_pu()
                else:
                    print("Invalid")
                    return
        else:
            return

    def clients_train_step_SL(self):
        if 'FedProx' in opt.method:
            print("Invalid")
            percentage = opt.percentage  # 0.5  0.9
            mu = opt.mu
            print(f"System heterogeneity set to {percentage}% stragglers.\n")
            print(f"Picking {len(self.clientSelect_idxs)} random clients per round.\n")
            heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients),
                                                          max_epochs=opt.local_epochs)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                  globalmodel=self.cloud.aggregated_client_model)
        elif 'FedAvg' in opt.method:
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                # self.clients[idx].train_fedavg_p()
                self.clients[idx].train_fedavg_p_mod()
        else:
            print("Invalid")
            return
