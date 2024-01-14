import torch
from copy import deepcopy
import torch.optim as optim
from pylab import *

from options import opt
from options import FedAVG_aggregated_model_path
from modules.loss import PLoss, MPULoss_V2, PLoss_my, PLoss_my2, PLoss_my3, PLoss_my4, PLoss_my5
from datasets.FMloader import DataLoader
from datasets.dataSpilt import CustomImageDataset, get_default_data_transforms


def adjust_learning_rate(optimizer, communication_round):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.pu_lr * (0.992 ** (communication_round * opt.local_epochs // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            self.loss = MPULoss_V2(opt.num_classes, opt.pu_weight).cuda()

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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_pu, step_size=1, gamma=0.992)
        self.optimizer_p = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler_p = optim.lr_scheduler.StepLR(self.optimizer_p, step_size=1, gamma=0.992)

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

                loss, puloss, celoss = self.loss(outputs, labels, self.priorlist, self.indexlist)
                # print("lr:", self.optimizer_pu.param_groups[-1]['lr'])
                loss.backward()
                total_loss.append(loss)
                self.optimizer_pu.step()
        print('mean loss of {} epochs: {:.4f}'.format(opt.local_epochs, (sum(total_loss) / len(total_loss)).item()))

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
            adjust_learning_rate(self.optimizer_pu, self.communicationRound)
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

    def train_fedavg_p_my2(self, global_logit):
        # print(global_logit) # 제대로 전달받음
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

        outputs_accumulator = {int(index.item()): [] for index in unique_indices}
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()

                for input, label in zip(inputs, labels):
                    if label.item() in unique_indices:
                        output = self.model(input.unsqueeze(0))
                        outputs_accumulator[label.item()].append(output)

        # counts = {} # 각각 120개씩 들어가는거 확인
        # for key in outputs_accumulator:
        #     counts[key] = len(outputs_accumulator[key])
        # print(counts)
        return outputs_accumulator
