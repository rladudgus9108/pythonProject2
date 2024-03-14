import torch
from copy import deepcopy
import torch.optim as optim
from pylab import *
import torch.nn as nn
import torch.nn.functional as F

from options import opt
from options import FedAVG_aggregated_model_path
from modules.loss import PLoss, MPULoss_V2, PLoss_my, PLoss_my2, PLoss_my3, PLoss_my4, PLoss_my5, MPULoss_my
from datasets.FMloader import DataLoader
from datasets.dataSpilt import CustomImageDataset, get_default_data_transforms


def adjust_learning_rate_my(optimizer, communication_round):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.pu_lr * (0.995 ** (communication_round * opt.local_epochs // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# default Setting
# def adjust_learning_rate(optimizer, communication_round):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = opt.pu_lr * (0.992 ** (communication_round * opt.local_epochs // 20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def calculate_distance(output, global_logit):
    distances = []
    for logit in global_logit:
        # 유클리드 거리 계산
        distance = torch.norm(output - logit, dim=1, p=2)  # default L2
        distances.append(distance.item())

    return distances


class Client:
    def __init__(self, client_id, model_pu, trainloader=None, testloader=None, priorlist=None, indexlist=None):
        self.client_id = client_id
        self.current_round = 0
        self.original_model = deepcopy(model_pu).cuda()
        self.model = model_pu
        if not opt.use_PULoss:
            self.loss = PLoss(opt.num_classes).cuda()
        else:
            # self.loss = MPULoss_INDEX(opt.num_classes, opt.pu_weight).cuda()
            self.loss = MPULoss_V2(opt.num_classes, opt.pu_weight).cuda()  # default
            # self.loss = MPULoss_my(opt.num_classes, opt.pu_weight).cuda()

        self.ploss = PLoss(opt.num_classes).cuda()
        self.ploss_my = PLoss_my(opt.num_classes).cuda()
        self.ploss_my2 = PLoss_my2(opt.num_classes).cuda()
        self.ploss_my3 = PLoss_my3(opt.num_classes).cuda()
        self.ploss_my4 = PLoss_my4(opt.num_classes).cuda()
        self.ploss_my5 = PLoss_my5(opt.num_classes).cuda()
        self.priorlist = priorlist
        self.indexlist = indexlist
        self.communicationRound = 0
        self.optimizer_pu = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_pu, step_size=1, gamma=0.995)  # default : 0.992
        self.optimizer_p = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler_p = optim.lr_scheduler.StepLR(self.optimizer_p, step_size=1, gamma=0.995)  # default : 0.992

        if not opt.useFedmatchDataLoader:
            self.train_loader = trainloader
            self.test_loader = testloader
        else:
            # for Fedmatch
            self.state = {'client_id': client_id}
            self.loader = DataLoader(opt)
            self.load_data()
            self.train_loader = self.getFedmatchLoader()

    def getFedmatchLoader(self):
        bsize_s = opt.bsize_s
        num_steps = round(len(self.x_labeled) / bsize_s)
        bsize_u = math.ceil(len(self.x_unlabeled) / max(num_steps, 1))  # 101

        self.y_labeled = torch.argmax(torch.from_numpy(self.y_labeled), -1).numpy()
        if 'SL' in opt.method:
            # make all the data full labeled
            self.y_unlabeled = torch.argmax(torch.from_numpy(self.y_unlabeled), -1).numpy()
        else:
            # sign the unlabeled data
            self.y_unlabeled = (torch.argmax(torch.from_numpy(self.y_unlabeled), -1) + opt.num_classes).numpy()

        # merge the S and U dataset
        train_x = np.concatenate((self.x_unlabeled, self.x_labeled), axis=0).transpose(0, 3, 1, 2)
        train_y = np.concatenate((self.y_unlabeled, self.y_labeled), axis=0)

        batchsize = bsize_s + bsize_u
        transforms_train, _ = get_default_data_transforms(opt.dataset, verbose=False)
        # train_dataset = CustomImageDataset(train_x, train_y, transforms_train)
        # Ablation
        train_dataset = CustomImageDataset(train_x.astype(np.float32) / 255, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

        return train_loader

    def load_original_model(self):
        self.model = deepcopy(self.original_model)
        self.communicationRound = 0
        self.optimizer_p = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler_p = optim.lr_scheduler.StepLR(self.optimizer_p, step_size=1, gamma=0.992)

    def initialize(self):
        if os.path.exists(FedAVG_aggregated_model_path):
            self.model.load_state_dict(torch.load(FedAVG_aggregated_model_path))

    def load_data(self):
        '''use FedMatch dataloader'''
        self.x_labeled, self.y_labeled, task_name = \
            self.loader.get_s_by_id(self.state['client_id'])
        self.x_unlabeled, self.y_unlabeled, task_name = \
            self.loader.get_u_by_id(self.state['client_id'], task_id=0)
        self.x_test, self.y_test = self.loader.get_test()
        self.x_valid, self.y_valid = self.loader.get_valid()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid)

    def train_fedavg_pu(self):
        self.model.train()
        total_loss = []

        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # print("training input img scale:", inputs.max(), inputs.min())
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_pu.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)
                print(self.model.state_dict())
                print("경계")
                loss, puloss, celoss = self.loss(outputs, labels, self.priorlist, self.indexlist)

                # print("lr:", self.optimizer_pu.param_groups[-1]['lr'])
                loss.backward()
                self.optimizer_pu.step()
                total_loss.append(loss)
                print(self.model.state_dict())
                print("경계")
        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # loss 계산을 self.loss()를 통해 하다보면 음수가 나올 수 있음
        self.communicationRound += 1
        self.scheduler.step()

    def train_fedavg_pu_mod(self):
        self.model.train()
        total_loss = []

        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # print("training input img scale:", inputs.max(), inputs.min())
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_pu.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)

                loss, puloss, celoss = self.loss(outputs, labels, self.priorlist, self.indexlist)
                # print("lr:", self.optimizer_pu.param_groups[-1]['lr'])
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_pu.step()
                    total_loss.append(loss)

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # loss 계산을 self.loss()를 통해 하다보면 음수가 나올 수 있음
        self.communicationRound += 1
        self.scheduler.step()

    def train_fedavg_pu_my(self):
        self.model.train()
        total_loss = []

        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # print("training input img scale:", inputs.max(), inputs.min())
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_pu.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)

                loss, puloss, celoss = self.loss(outputs, labels, self.priorlist, self.indexlist)

                # print("lr:", self.optimizer_pu.param_groups[-1]['lr'])
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_pu.step()
                    total_loss.append(loss)

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # loss 계산을 self.loss()를 통해 하다보면 음수가 나올 수 있음
        self.communicationRound += 1
        self.scheduler.step()

    def train_fedavg_pu_my_2(self, global_logit):
        self.model.train()
        total_loss = []

        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # print("training input img scale:", inputs.max(), inputs.min())
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_pu.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)

                loss, puloss, celoss, reloss = self.loss(outputs, labels, self.priorlist, self.indexlist, global_logit)
                # loss : 최종 loss, puloss : pu1 + pu2 + pu3, celoss : label된 데이터, reloss : pu4
                # print("lr:", self.optimizer_pu.param_groups[-1]['lr'])
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_pu.step()
                    total_loss.append(loss)

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # loss 계산을 self.loss()를 통해 하다보면 음수가 나올 수 있음
        self.communicationRound += 1
        self.scheduler.step()

    def train_fedprox_p(self, epochs=20, mu=0.0, globalmodel=None):
        self.model.train()
        total_loss = []
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # print("training input img scale:", inputs.max(), inputs.min())
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)

                loss = self.ploss(outputs, labels)

                proximal_term = torch.zeros(1).cuda()
                # iterate through the current and global model parameters
                for w, w_t in zip(self.model.state_dict().items(), globalmodel.state_dict().items()):
                    if (w[1] - w_t[1]).dtype == torch.float:
                        proximal_term += (w[1] - w_t[1]).norm(2)
                loss = loss + (mu / 2) * proximal_term

                loss.backward()
                total_loss.append(loss)
                self.optimizer_p.step()
        print('mean loss of {} epochs: {:.4f}'.format(epoch, (sum(total_loss) / len(total_loss)).item()))

        self.communicationRound += 1
        self.scheduler_p.step()

    def train_fedprox_pu(self, epochs=20, mu=0.0, globalmodel=None):
        self.model.train()
        total_loss = []
        if opt.adjust_lr:
            adjust_learning_rate_my(self.optimizer_pu, self.communicationRound)
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # print("training input img scale:", inputs.max(), inputs.min())
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_pu.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)

                loss, puloss, celoss = self.loss(outputs, labels, self.priorlist, self.indexlist)

                proximal_term = 0.0
                # iterate through the current and global model parameters

                if globalmodel == None:
                    globalmodel = self.model

                for w, w_t in zip(self.model.state_dict().items(), globalmodel.state_dict().items()):
                    # update the proximal term
                    # proximal_term += torch.sum(torch.abs((w-w_t)**2))
                    if (w[1] - w_t[1]).dtype == torch.float:
                        proximal_term += (w[1] - w_t[1]).norm(2)

                loss = loss + (mu / 2) * proximal_term
                total_loss.append(loss)
                loss.backward()
                self.optimizer_pu.step()
            # print("epoch", epoch, "lr:", self.optimizer_pu.state_dict()['param_groups'][0]['lr'])
        print('mean loss of {} epochs: {:.4f}'.format(epoch, (sum(total_loss) / len(total_loss)).item()))
        self.communicationRound += 1
        self.scheduler.step()

    def train_fedavg_p(self):
        self.model.train()
        total_loss = []
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                loss = self.ploss(outputs, labels)
                loss.backward()
                self.optimizer_p.step()
                total_loss.append(loss)
        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    def train_fedavg_p_mod(self):
        self.model.train()
        total_loss = []
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                loss = self.ploss(outputs, labels)

                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_p.step()
                    total_loss.append(loss)
        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    def train_fedavg_p_mod_1(self):
        self.model.train()
        self.numClass = 10
        total_loss = []
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                outputs = outputs.cuda().float()
                # P_mask 는 label이 되어 있는 데이터의 인덱스를 나타냄
                P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
                labelsP = torch.index_select(labels, 0, P_mask).cuda()
                outputsP = torch.index_select(outputs, 0, P_mask).cuda()

                crossentropyloss = nn.CrossEntropyLoss().cuda()

                loss = crossentropyloss(outputsP, labelsP)

                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_p.step()
                    total_loss.append(loss)
        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    def train_fedavg_p_mod_2(self, global_logit):
        self.model.train()
        self.numClass = 10

        total_loss = []
        alphaWeight = 0.5  # loss의 가중치
        betaWeight = 0.5

        # print(self.model.state_dict())
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                outputs = outputs.cuda().float()

                crossentropyloss = nn.CrossEntropyLoss().cuda()

                P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
                labelsP = torch.index_select(labels, 0, P_mask).cuda()
                outputsP = torch.index_select(outputs, 0, P_mask).cuda()
                # 이 부분에 outputsP에 대해서 softmax를 취한 다음에 loss 를 구하게끔 수정하는 것
                # 해볼만한 부분임

                alphaLoss = crossentropyloss(outputsP, labelsP)

                U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
                outputsU = torch.index_select(outputs, 0, U_mask).cuda()

                # 거리 계산
                similarites = self.calculate_similarities(global_logit, outputsU)

                max_similarities, max_indices = similarites.max(dim=1)

                # 상위 10% 거리에 해당하는 인덱스를 선택
                num_selected = int(len(max_similarities) * 0.1)  # 10%
                top_distances_indices = max_similarities.topk(num_selected, largest=True)[1]

                # 상위 10%에 해당하는 outputs와 labels만 선택
                selected_outputs = outputsU[top_distances_indices]
                selected_labels = max_indices[top_distances_indices]

                # 이 부분에도 마찬가지로 selected_outputs에 대해서 softmax 를 취한 다음에 loss 계산하는 것 그 부분 반영해보기
                betaLoss = crossentropyloss(selected_outputs, selected_labels)

                loss = alphaWeight * alphaLoss + betaWeight * betaLoss

                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_p.step()
                    total_loss.append(loss)
        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함
        print("_________________________________________")
        # print(self.model.state_dict())
        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    def train_fedavg_p_mod_3(self, global_logit):
        self.model.train()
        self.numClass = 10

        total_loss = []
        # print(self.model.state_dict())
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                outputs = outputs.cuda().float()

                P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
                labelsP = torch.index_select(labels, 0, P_mask).cuda()
                outputsP = torch.index_select(outputs, 0, P_mask).cuda()

                U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
                outputsU = torch.index_select(outputs, 0, U_mask).cuda()

                # 거리 계산
                similarites = self.calculate_similarities(global_logit, outputsU)

                max_similarities, max_indices = similarites.max(dim=1)

                # 상위 10% 거리에 해당하는 인덱스를 선택
                num_selected = int(len(max_similarities) * 0.1)  # 10%
                top_distances_indices = max_similarities.topk(num_selected, largest=True)[1]

                # 상위 10%에 해당하는 outputs와 labels만 선택
                selected_outputs = outputsU[top_distances_indices]
                selected_labels = max_indices[top_distances_indices]

                # 선택된 outputs와 labels를 기존의 outputsP와 labelsP에 추가
                labels_combined = torch.cat([labelsP, selected_labels], dim=0)
                outputs_combined = torch.cat([outputsP, selected_outputs], dim=0)

                crossentropyloss = nn.CrossEntropyLoss().cuda()
                loss = crossentropyloss(outputs_combined, labels_combined)

                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer_p.step()
                    total_loss.append(loss)
        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함
        print("_________________________________________")
        # print(self.model.state_dict())
        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    def calculate_similarities(self, logits, outputs):
        similarities = []
        for logit in logits:
            similarity = F.cosine_similarity(logit, outputs, dim=1)
            similarities.append(similarity)
        return torch.stack(similarities, dim=1)

    def train_fedavg_p_my2(self, global_logit):
        # print(global_logit) # 제대로 전달받음
        # print(self.model.state_dict())
        self.model.train()
        total_loss = []
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                loss = self.ploss_my2(outputs, labels, global_logit)
                if not torch.isnan(loss):
                    # 여기서 loss가 줄어들지만 이걸 server로 모았을때 결과가 이상함
                    loss.backward()
                    self.optimizer_p.step()
                    total_loss.append(loss)
                print("_______________________")
        # print(self.model.state_dict())

        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    # my 를 먼저 구현한거임 이거는 calculate_distance 하는 부분을 적용한 함수
    def train_fedavg_p_my(self, global_logit):
        new_global_logit = global_logit
        # print(new_global_logit) # 제대로 전달받음 여기서 확인하고 싶은거는
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                for input, label in zip(inputs, labels):
                    if label.item() >= 10:
                        total += 1
                        input = input.unsqueeze(0)
                        output = self.model(input)
                        distances = calculate_distance(output, new_global_logit)
                        pred_label = distances.index(min(distances))
                        if (pred_label == (label.item() % 10)):
                            correct += 1

            print('Accuracy of the client{} on relabeling the unlabel training sets: {:.4f} %'.format(self.client_id,
                                                                                                      100 * correct / total))

        self.model.train()
        total_loss = []
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # import pdb; pdb.set_trace()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # outputs 는 일단 다 구하고 loss를 구할때만 labeling으로 구분된 거만 반영
                loss = self.ploss(outputs, labels)
                loss.backward()
                self.optimizer_p.step()
                total_loss.append(loss)
        # total_loss_sum은 내가 수정해서 구현한 부분
        # default : print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))
        # 하지만 여기 출력 부분이 20 epoch마다 loss 평균이 얼마가 되는지 출력하는 부분으로 굳이 출력해야 되나 생각하여 주석 처리함

        total_loss_sum = sum(tensor.cpu().item() for tensor in total_loss)
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, total_loss_sum / len(total_loss)))

        self.communicationRound += 1
        self.scheduler_p.step()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs)
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            total += pred.size(0)
            correct += (pred == labels).sum().item()
        print('Accuracy of the {} on the testing sets: {:.4f} %%'.format(self.client_id, 100 * correct / total))
        return 100 * correct / total

    def send_logit(self):
        self.model.eval()
        unique_indices = torch.nonzero(self.indexlist).flatten()
        temperature = 0.1
        outputs_accumulator = {int(index.item()): [] for index in unique_indices}
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()

                for input, label in zip(inputs, labels):
                    if label.item() in unique_indices:
                        output = self.model(input.unsqueeze(0))
                        output_softmax = F.softmax(output / temperature, dim=1)

                        outputs_accumulator[label.item()].append(output_softmax)

        # counts = {} # 각각 120개씩 들어가는거 확인
        # for key in outputs_accumulator:
        #     counts[key] = len(outputs_accumulator[key])
        # print(counts)
        return outputs_accumulator

    def send_logit_std(self):
        self.model.eval()
        unique_indices = torch.nonzero(self.indexlist).flatten()
        # temperature = 1.0
        outputs_accumulator = {int(index.item()): [] for index in unique_indices}
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()

                for input, label in zip(inputs, labels):
                    if label.item() in unique_indices:
                        output = self.model(input.unsqueeze(0))
                        output_std = output.std().item()
                        output_softmax = F.softmax(output / output_std, dim=1)

                        outputs_accumulator[label.item()].append(output_softmax)

        # counts = {} # 각각 120개씩 들어가는거 확인
        # for key in outputs_accumulator:
        #     counts[key] = len(outputs_accumulator[key])
        # print(counts)
        return outputs_accumulator

    def send_logit_range(self):
        self.model.eval()
        unique_indices = torch.nonzero(self.indexlist).flatten()
        area = 5.0  # 0 : 실행되면 안됨, 1 : 0.1, 2 : 0.1, 0.2
        outputs_accumulator = {int(index.item()): [] for index in unique_indices}
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()

                for input, label in zip(inputs, labels):
                    if label.item() in unique_indices:
                        output = self.model(input.unsqueeze(0))
                        output_range = torch.zeros(1, 10).cuda()
                        for i in range(int(area)):
                            output_softmax = F.softmax(output / ((i + 1) / 10), dim=1)
                            output_range += output_softmax

                        output_range = output_range / area

                        outputs_accumulator[label.item()].append(output_range)

        # counts = {} # 각각 120개씩 들어가는거 확인
        # for key in outputs_accumulator:
        #     counts[key] = len(outputs_accumulator[key])
        # print(counts)
        return outputs_accumulator
