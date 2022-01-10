
"""
Cloud server
"""
import torch
import copy
import torch.nn as nn
from options import FedAVG_model_path, FedAVG_aggregated_model_path
from torch.utils.data import DataLoader


class Cloud:
    def __init__(self, clients, model, numclasses, dataloader):
        self.model = model
        self._save_model()
        self.clients = clients
        self.numclasses = numclasses
        self.test_loader = dataloader
        self.participating_clients = None
        self.aggregated_client_model = None

    def aggregate(self, clientSelect_idxs):
        totalsize = 0
        samplesize = 500
        for idx in clientSelect_idxs:
            totalsize += samplesize

        for k, idx in enumerate(clientSelect_idxs):
            client = self.clients[idx]
            weight = samplesize / totalsize

            for name, param in client.model.state_dict().items():
                if k == 0:
                    self.aggregated_client_model[name] = param.data * weight
                else:
                    self.aggregated_client_model[name] += param.data * weight

        return self.aggregated_client_model

    def validation(self, cur_rounds):
        self.model.eval()
        correct = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            # print("Test input img scale:", inputs.max(), inputs.min())
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs)
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            correct += (pred == labels).sum().item()
        print('Round:{:d}, Accuracy: {:.4f} %'.format(cur_rounds, 100 * correct / len(self.test_loader.dataset)))
        return 100 * correct / len(self.test_loader.dataset)



    def _save_model(self):
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)

