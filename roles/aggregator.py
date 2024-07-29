"""
Cloud server
"""
import torch
import copy
import torch.nn as nn
from options import FedAVG_model_path, FedAVG_aggregated_model_path
from torch.utils.data import DataLoader
from collections import defaultdict


class Cloud:
    def __init__(self, clients, model, numclasses, dataloader):
        self.model = model
        self._save_model()
        self.clients = clients
        self.numclasses = numclasses
        self.test_loader = dataloader
        self.participating_clients = None
        self.aggregated_client_model = model

    def aggregate(self, clientSelect_idxs):
        totalsize = 0
        samplesize = 500 # sampleSize가 뭘까
        for idx in clientSelect_idxs:
            totalsize += samplesize

        global_model = {}
        model_dict = self.aggregated_client_model.state_dict()
        for k, idx in enumerate(clientSelect_idxs):
            client = self.clients[idx]
            weight = samplesize / totalsize
            for name, param in client.model.state_dict().items():
                if k == 0:
                    global_model[name] = param.data * weight
                else:
                    global_model[name] += param.data * weight

        pretrained_dict = {k: v for k, v in global_model.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.aggregated_client_model.load_state_dict(pretrained_dict)
        return self.aggregated_client_model

    def validation(self, cur_rounds):
        self.aggregated_client_model.eval()
        correct = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            # print("Test input img scale:", inputs.max(), inputs.min())
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.aggregated_client_model(inputs)
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            correct += (pred == labels).sum().item()
        # 만약 여기라면 내가 원하는 것은 가장 최고일때의 정확도가 몇이며, 그때의 인덱스가 무엇인지 출력하는 부분을 구현
        # 여기가 맞음
        print(
            'Current Round:{:d}, Accuracy: {:.4f} %'.format(cur_rounds, 100 * correct / len(self.test_loader.dataset)))
        return 100 * correct / len(self.test_loader.dataset)

    def new_validation(self, cur_rounds):
        self.aggregated_client_model.eval()
        correct = 0
        label_correct = defaultdict(int)
        label_total = defaultdict(int)

        for i, (inputs, labels) in enumerate(self.test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.aggregated_client_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    label_correct[label.item()] += 1
                label_total[label.item()] += 1

        total_accuracy = 100 * correct / len(self.test_loader.dataset)
        print('Current Round: {:d}, Total Accuracy: {:.4f} %'.format(cur_rounds, total_accuracy))

        # 각 라벨별 정확도 출력
        for label in sorted(label_correct.keys()):
            label_accuracy = 100 * label_correct[label] / label_total[label]
            print('Label {:d} Accuracy: {:.4f} %'.format(label, label_accuracy))

        return total_accuracy

    def _save_model(self):
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)

    # def calculate_global_logit(self, tensor_dict_list): # 평균내서 받은 logit으로 가중 평균 합 내는 방법
    #     tensor_sums = {}
    #     tensor_counts = {}
    #
    #     # 모든 딕셔너리를 순회하며 텐서들을 더함
    #     for tensor_dict in tensor_dict_list:
    #         for key, tensors in tensor_dict.items():
    #             if key not in tensor_sums:
    #                 tensor_sums[key] = tensors[0] * tensors[1]
    #                 tensor_counts[key] = tensors[1]
    #             else:
    #                 tensor_sums[key] += tensors[0] * tensors[1]
    #                 tensor_counts[key] += tensors[1]
    #
    #     # 각 키에 대한 평균 텐서 계산
    #     average_tensors = {key: tensor_sums[key] / tensor_counts[key] for key in tensor_sums}
    #
    #     # 결과 리스트 생성
    #     result = []
    #     for i in average_tensors.keys():
    #         result.append(average_tensors.get(i, []))
    #
    #     return result

    def calculate_global_logit(self, tensor_dict_list):
        tensor_sums = {}
        tensor_counts = {}

        # 모든 딕셔너리를 순회하며 텐서들을 더함
        for tensor_dict in tensor_dict_list:
            for key, tensors in tensor_dict.items():
                if key not in tensor_sums:
                    tensor_sums[key] = sum(tensors)
                    tensor_counts[key] = len(tensors)
                else:
                    tensor_sums[key] += sum(tensors)
                    tensor_counts[key] += len(tensors)

        # 각 키에 대한 평균 텐서 계산
        average_tensors = {key: tensor_sums[key] / tensor_counts[key] for key in tensor_sums}

        # 결과 리스트 생성
        result = []
        for i in average_tensors.keys():
            result.append(average_tensors.get(i, []))

        return result
