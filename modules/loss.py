import torch
import torch.nn as nn
import torch.nn.functional as F


class MPULoss(nn.Module):
    def __init__(self, k, PiW, PkW, UiW, UkW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.piW = PiW
        self.pkW = PkW
        self.uiW = UiW
        self.ukW = UkW

    def forward(self, outputs, labels, prior):
        outputs = outputs.cuda().float()
        outputs_Soft = F.softmax(outputs, dim=1)
        # 数据划分
        P_mask = (labels < self.numClass - 1).nonzero().view(-1)
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels >= self.numClass - 1).nonzero().view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask)
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)

        # 计算目标
        ui = sum(-torch.log(1 - outputsU_Soft[:, 0:self.numClass - 1] + 0.01)) / (
                (self.numClass - 1) * outputsU.size(0))
        uk = sum(-torch.log(outputsU_Soft[:, self.numClass - 1] + 0.01)) / outputsU.size(0)

        UnlabeledLossI = sum(ui)
        UnlabeledLossK = uk

        crossentropyloss = nn.CrossEntropyLoss()
        PositiveLossI = crossentropyloss(outputsP, labelsP)
        PositiveLossK = sum(-torch.log(1 - outputsP_Soft[:, self.numClass - 1] + 0.01)) * prior

        objective = PositiveLossI * self.piW + PositiveLossK * self.pkW + UnlabeledLossI * self.uiW + UnlabeledLossK * self.ukW  # w将三者统一到同一个数量级上
        # print("\n 1")
        # print(PositiveLossI*self.piW, PositiveLossK*self.pkW)
        # print(UnlabeledLossI*self.uiW, UnlabeledLossK*self.ukW)
        return objective, PositiveLossI * self.piW + PositiveLossK * self.pkW, UnlabeledLossI * self.uiW + UnlabeledLossK * self.ukW


class MPULoss_INDEX(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW

    def forward(self, outputs, labels, priorlist, indexlist):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        # 数据划分
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)
        # import pdb; pdb.set_trace()

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        outputsU = torch.index_select(outputs, 0, U_mask)  # unlabeldata 的 ground truth. setting限制不能使用
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)  # 所有在 unlabeldata 上的预测值

        PULoss = torch.zeros(1).cuda()

        for i in range(self.numClass):
            if i in indexlist:  # calculate ui
                pu3 = sum(-torch.log(1 - outputsU_Soft[:, i] + 0.01)) / \
                      max(1, outputsU.size(0)) / len(indexlist)
                PULoss += pu3
            else:
                pu1 = sum(-torch.log(1 - outputsP_Soft[:, i] + 0.01)) * \
                      priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass - len(indexlist))
                PULoss += pu1

        pu2 = torch.zeros(1).cuda()
        for index, i in enumerate(labelsP):  # need to be optimized
            x = outputsP_Soft[index][i]
            pu2 += -torch.log(1 - x + 0.01) * priorlist[i]

        PULoss -= pu2 / max(1, outputsP.size(0))

        crossentropyloss = nn.CrossEntropyLoss()
        crossloss = crossentropyloss(outputsP, labelsP)

        # objective = PULoss * self.puW
        objective = crossloss
        # objective = PULoss * self.puW + crossloss

        return objective, PULoss * self.puW, crossloss


class PLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    def forward(self, outputs, labels):
        outputs = outputs.cuda().float()
        # P_mask 는 label이 되어 있는 데이터의 인덱스를 나타냄
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).cuda()
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        crossentropyloss = nn.CrossEntropyLoss().cuda()

        crossloss = crossentropyloss(outputsP, labelsP)
        return crossloss


class MPULoss_V2(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW
        # puW : weight of puloss

    def forward(self, outputs, labels, priorlist, indexlist):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        new_P_indexlist = indexlist
        torch.zeros(self.numClass).cuda()
        eps = 1e-6

        # P U data
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        outputsU = torch.index_select(outputs, 0, U_mask)
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)

        PULoss = torch.zeros(1).cuda()

        pu3 = (-torch.log(1 - outputsU_Soft + eps) * new_P_indexlist).sum() / max(1, outputsU.size(0)) / len(
            indexlist)  # pU loss

        PULoss += pu3
        if self.numClass > len(indexlist):  # 반복문에 들어가지 않음 pu1은 계산되지 않음
            # numClass : 10, len(indexlist) : 10
            pu1 = (-torch.log(1 - outputsP_Soft + eps) * new_P_indexlist).sum() * \
                  priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass - len(indexlist))
            PULoss += pu1

        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass * 2).cuda().scatter_(1, torch.unsqueeze(labelsP, 1),
                                                                                         1)[:, :self.numClass]
        log_res = -torch.log(1 - outputsP_Soft * label_onehot_P + eps)
        pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
        # default : pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))

        PULoss += pu2
        # 무조건 pu2에 의해서 loss가 음수가 나옴

        crossentropyloss = nn.CrossEntropyLoss()
        crossloss = crossentropyloss(outputsP, labelsP)

        if crossloss.isnan():
            objective = 1 * PULoss
        else:
            objective = 1 * PULoss * self.puW + crossloss * 1

        # objective = abs(PULoss * self.puW + crossloss) 절대값을 취해서 실행도 시켜보는중
        # loss 계산에 따라서 더해지는 값이 음수가 크게 반환될 경우 total_loss에 저장되는 loss가 음수일수 있다
        # 그 때를 방지하고자 loss가 아예 음수가 되는 것을 여기서 방지하고 들어가는 것을 실행중
        # 음수를 만드는 거는 PULoss에 의해서 생기는 거임

        return objective, PULoss * self.puW, crossloss


class MPULoss_suggest(nn.Module):  # 여기만 고쳐서 진행하면됨
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW

    def calculate_similarities(self, logits, outputs):
        similarities = []
        for logit in logits:
            similarity = F.cosine_similarity(logit, outputs, dim=1)
            similarities.append(similarity)
        return torch.stack(similarities, dim=1)

    def forward(self, outputs, labels, priorlist, indexlist, global_logit, percentage_of_relabeling_rate):
        puRate = 2
        crossRate = 2
        reRate = 1
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        new_P_indexlist = indexlist
        torch.zeros(self.numClass).cuda()
        eps = 1e-6

        # Positive 데이터와 Unlabeled 데이터 분리
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        outputsU = torch.index_select(outputs, 0, U_mask)  # softmax 적용 전의 값을 사용

        PULoss = torch.zeros(1).cuda()

        # pu4 손실 계산을 위한 거리 계산
        distances = self.calculate_similarities(global_logit, outputsU)
        # 각 Unlabeled 데이터에 대해 거리 계산 후 가장 유사도가 높은 클래스와 유사도 값 구하기
        max_similarities, max_indices = distances.max(dim=1)

        # 유사도가 높은 상위 10%의 데이터 인덱스 선택, 배치 안의 상위 x%임
        num_selected = int(len(outputs) * percentage_of_relabeling_rate)
        _, top_sim_indices = max_similarities.topk(num_selected, largest=True)

        # 상위 10% 데이터에 대한 임시 라벨 할당
        selected_U_mask = top_sim_indices
        selected_labels = max_indices[selected_U_mask]

        crossentropyloss = nn.CrossEntropyLoss()
        selected_outputsU = torch.index_select(outputsU, 0, selected_U_mask)

        pu4_loss = crossentropyloss(selected_outputsU, selected_labels)

        # "--------------------------------------------------------------------"
        # 나머지 Unlabeled 데이터에 대한 pu3 손실 계산
        remaining_U_mask = [i for i in range(len(outputsU)) if i not in selected_U_mask]
        remaining_outputsU = torch.index_select(outputsU, 0, torch.tensor(remaining_U_mask).cuda())
        remaining_outputsU_Soft = F.softmax(remaining_outputsU, dim=1)

        pu3_loss = (-torch.log(1 - remaining_outputsU_Soft + eps) * new_P_indexlist).sum() / max(1,
                                                                                                 remaining_outputsU.size(
                                                                                                     0)) / len(
            indexlist)
        PULoss += pu3_loss

        # pu1 손실 계산
        if self.numClass > len(indexlist):  # numClass : 10, len(indexlist) : 10 으로 반복문에 들어가지 않음
            pu1_loss = (-torch.log(1 - outputsP_Soft + eps) * new_P_indexlist).sum() * \
                       priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass - len(indexlist))
            PULoss += pu1_loss

        # pu2 손실 계산 (Label된 데이터)
        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass * 2).cuda().scatter_(1, torch.unsqueeze(labelsP, 1),
                                                                                         1)[:, :self.numClass]
        log_res = -torch.log(1 - outputsP_Soft * label_onehot_P + eps)
        pu2_loss = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
        PULoss += pu2_loss

        # crossloss 계산 (Positive 데이터에 대한 교차 엔트로피 손실)
        crossloss = crossentropyloss(outputsP, labelsP)

        if crossloss.isnan():  # Prof Yang : 여기에서 배치 안에 레이블된 데이터가 적으니깐 비율을 조정해서 하는 방법을 생각해봐라 조언
            objective = puRate * PULoss * self.puW + reRate * pu4_loss
        else:
            objective = puRate * PULoss * self.puW + crossRate * crossloss + reRate * pu4_loss

        return objective, PULoss * self.puW, crossloss, pu4_loss
