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


class FmpuTrainer:
    def __init__(self, model_pu):
        # load data
        if not opt.useFedmatchDataLoader:  # default 로 usedFedmatchDatLoader : False가 되어 있음
            # create Clients and Aggregating Server
            local_dataloaders, local_sample_sizes, test_dataloader, indexlist, priorlist = get_data_loaders()
            # 윗줄에서 client별 들어가는 class와 거기서 select되는 class들 정해지는게 다 처리됨

            self.clients = [Client(_id + 1, copy.deepcopy(model_pu).cuda(), local_dataloaders[_id], test_dataloader,
                                   priorlist=priorList, indexlist=indexList)
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

        for t in range(self.communication_rounds):
            self.current_round = t + 1
            self.cloud_lastmodel = self.cloud.aggregated_client_model
            self.clients_select()  # random choice로 clients select, but 1.0으로 되어 있어서 모두 다 선택됨

            if 'SL' in opt.task:  # default : opt.method 라고 되어 있었음
                print("##### Full labeled setting #####")
                self.clients_train_step_SL()
            else:
                print("##### Semi-supervised setting #####")
                self.clients_train_step_SS()  # memery up

            self.cloud.aggregate(self.clientSelect_idxs)
            acc_per_round.append(self.cloud.validation(t))

            max_acc = max(acc_per_round)
            max_acc_round = acc_per_round.index(max_acc)
            print("Hightest Round:{}, Highest Accuracy: {:.4f} %".format(max_acc_round, max_acc))
            print("__________________________________________")

    def clients_select(self):
        m = max(int(opt.clientSelect_Rate * opt.num_clients), 1)
        # default로 설정된 것은 아래 코드
        # self.clientSelect_idxs = np.random.choice(range(opt.num_clients), m, replace=False)
        self.clientSelect_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # default
        # self.clientSelect_idxs = [0, 1, 2, 3, 4]
        # self.clientSelect_idxs = [0, 1]
        # self.clientSelect_idxs = [0 ,1, 2, 3, 4, 5, 6, 7, 8]

    def clients_train_step_SS(self):
        if 'FedProx' in opt.method:
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
                    self.clients[idx].train_fedavg_pu_mod()  # opt에 PULoss를 설정하는 부분은 없음 -> config.py에서 설정함
                else:  # 내가 지금 생각하는 것은 fedavg_p()를 통해서 positive data로만 loss를 구하는 부분에 있어서
                    # 이 부분에서 unlabel data에 대해서 라벨링을 해서 어떻게 나오는지까지만을 보여주면 되지 않을까
                    # PULoss에 적용하기에는 이 논문에 대해서 제대로 분석이 이뤄지지 않았기 때문에 어느 부분에 이것을 적용해야할지 모르겠음
                    self.clients[idx].train_fedavg_p()
        else:
            return

    def clients_train_step_SS_my(self):
        if 'FedProx' in opt.method:
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
                    self.clients[idx].train_fedavg_pu()  # opt에 PULoss를 설정하는 부분은 없음 -> config에서 설정
                    # default : self.clients[idx].train_fedavg_pu()
                else:  # 내가 지금 생각하는 것은 fedavg_p()를 통해서 positive data로만 loss를 구하는 부분에 있어서
                    # 이 부분에서 unlabel data에 대해서 라벨링을 해서 어떻게 나오는지까지만을 보여주면 되지 않을까
                    # PULoss에 적용하기에는 이 논문에 대해서 제대로 분석이 이뤄지지 않았기 때문에 어느 부분에 이것을 적용해야할지 모르겠음
                    self.clients[idx].train_fedavg_p_mod()
                    # 원래 fedavg_p() 였는데 수정함
        else:
            return

    def clients_train_step_SS_my_2(self):  # 현재 이 코드로 실행중
        if 'FedProx' in opt.method:
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
            client_logit_store = []
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                client_logit_store.append(self.clients[idx].send_logit())

            global_logit = self.cloud.calculate_global_logit(client_logit_store)

            for idx in self.clientSelect_idxs:
                if opt.use_PULoss:
                    self.clients[idx].train_fedavg_pu_my_2(global_logit)
                    # default : self.clients[idx].train_fedavg_pu()
                else:
                    self.clients[idx].train_fedavg_p_mod()
                    # default : self.clients[idx].train_fedavg_p()
        else:
            return

    def clients_train_step_SL(self):
        if 'FedProx' in opt.method:
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
                self.clients[idx].train_fedavg_p_mod()
        else:
            return

    def clients_train_step_SL_my(self):
        if 'FedProx' in opt.method:
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
            client_logit_store = []
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                client_logit_store.append(self.clients[idx].send_logit())

            global_logit = self.cloud.calculate_global_logit(client_logit_store)
            for idx in self.clientSelect_idxs:
                self.clients[idx].train_fedavg_p_mod_2(global_logit)
        else:
            return

    # def clients_train_step_SL_my_round(self):
    #     if 'FedProx' in opt.method:
    #         percentage = opt.percentage  # 0.5  0.9
    #         mu = opt.mu
    #         print(f"System heterogeneity set to {percentage}% stragglers.\n")
    #         print(f"Picking {len(self.clientSelect_idxs)} random clients per round.\n")
    #         heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients),
    #                                                       max_epochs=opt.local_epochs)
    #         heterogenous_epoch_list = np.array(heterogenous_epoch_list)
    #         for idx in self.clientSelect_idxs:
    #             self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
    #             self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
    #                                               globalmodel=self.cloud.aggregated_client_model)
    #     elif 'FedAvg' in opt.method:
    #         if self.communication_rounds % 5 == 1:
    #             client_logit_store = []
    #             for idx in self.clientSelect_idxs:
    #                 print(self.current_round)
    #                 self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
    #                 client_logit_store.append(self.clients[idx].send_logit())
    #             global_logit = self.cloud.calculate_global_logit(client_logit_store)
    #         else :
    #             global_logit = self.previous_global_logit
    #
    #         for idx in self.clientSelect_idxs:
    #             self.clients[idx].train_fedavg_p_mod_2(global_logit)
    #     else:
    #         return
