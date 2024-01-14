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


class PLoss_my(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    """
            그럼 이제 해야 되는게 P_mask에 해당하지 않는 인덱스들에 대해서
            거기에 해당하는 인덱스들의 output 값들에 대해 calculate_distance()를 진행하여 label을 예측한다.
            그리고 거리가 작게 나온 상위 10%를 선택한 후
            그 때의 label은 calculate_distance()를 하여 제일 작게 나온 인덱스로 설정하고
            그 때의 output은 기존의 outputs에서 가져올 수 있도록 한다
            그렇게 되면 얻은 label과 output을 각각 구할 수 있고 
            그것을 labelsP와 outputsP와 합친 새로운 것을 얻을 수 있다.
            그 다음에 crossentropyloss를 구하면 되는 것이다.
            #### 여기서 주의 해야될 점은 label을 할 때 기존의 labels - 10을 하는 것이 아니라 calculate_distance()를 통해 얻은 최소 거리의
            인덱스로 label을 저장하는 것이다. 즉 기존의 label을 건드리지 않고 새로운 변수에다가 저장해놓는 것이다. ####
            ### 추가로 output을 index_select()를 통하여 불러올텐데 그때 값이 제대로 가져와 지는지를 확인해야 한다.
            # 어떻게 보면 loss 만을 구할때 해당하는 output에 대해서 임시로 labeling을 하여 참가 시키는 구조인 것이다.
            ### 이것을 구현하면 됨
            # 이것을 구현했다 하면 다음으로 생각할 수 있는 것은
            # 1. calculate_distance()를 할때, 코사인 유사도를 사용하여 계산하는 방법
            # 2. 거리계산 / 코사인 유사도 를 적용하지만, 각각 distance 혹은 similarity를 return할 때의 계산 방식에 차이를 두는 것이다
            # 2-1. 코사인 유사도의 경우 dist2 / dist1 을 생각할 수 있다
            # 2-2. 코사인 유사도의 경우 dist1 - dist2 를 적용할 수 있다
            # 3. calculate_distnace()를 할 때 계산하는 방식에 변화를 적용할 수 있다.
            # 3-1. 
            """

    def forward(self, outputs, labels, global_logit):
        # print(global_logit)
        outputs = outputs.cuda().float()
        # 이 부분에서 global_logit을 가지고 계산을 해주면 될듯
        # P_mask 는 label이 되어 있는 데이터의 인덱스를 나타냄
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)  # labels < 10인 인덱스 return
        labelsP = torch.index_select(labels, 0, P_mask).cuda()  # 그 때 해당하는 label이 무엇인지 return
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask).cuda()

        distances = self.calculate_distances(global_logit, outputsU)
        min_distance_labels = distances.min(dim=1)[1]

        labels_combined = torch.cat([labelsP, min_distance_labels], dim=0)
        outputs_combined = torch.cat([outputsP, outputsU], dim=0)

        crossentropyloss = nn.CrossEntropyLoss().cuda()

        crossloss = crossentropyloss(outputs_combined, labels_combined)
        return crossloss

    def calculate_distances(self, logits, outputs):
        # 각 로짓과 아웃풋 사이의 거리를 계산합니다.
        distances = []
        for logit in logits:
            distance = torch.norm(logit - outputs, dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)


class PLoss_my2(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    def forward(self, outputs, labels, global_logit):
        outputs = outputs.cuda().float()
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).cuda()
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask).cuda()

        # 거리 계산
        distances = self.calculate_distances(global_logit, outputsU)

        min_distances, min_indices = distances.min(dim=1)
        # min_indices = distances.min(dim=1)[1]
        # 상위 10% 거리에 해당하는 인덱스를 선택
        num_selected = int(len(min_distances) * 0.1)  # 10%
        top_distances_indices = min_distances.topk(num_selected, largest=False)[1]

        # 상위 10%에 해당하는 outputs와 labels만 선택
        selected_outputs = outputsU[top_distances_indices]
        selected_labels = min_indices[top_distances_indices]

        # 선택된 outputs와 labels를 기존의 outputsP와 labelsP에 추가
        labels_combined = torch.cat([labelsP, selected_labels], dim=0)
        outputs_combined = torch.cat([outputsP, selected_outputs], dim=0)

        crossentropyloss = nn.CrossEntropyLoss().cuda()
        crossloss = crossentropyloss(outputs_combined, labels_combined)
        return crossloss

    def calculate_distances(self, logits, outputs):
        distances = []
        for logit in logits:
            distance = torch.norm(logit - outputs, dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)

    def calculate_similarities(self, logits, outputs):
        similarities = []
        # for logit in logits :
        # similarity =
        # similarities.append(similarity)
        # return


class PLoss_my3(nn.Module):  # cosine 유사도 계산 + 상위 10 % 구현
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    def forward(self, outputs, labels, global_logit):
        outputs = outputs.cuda().float()
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).cuda()
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask).cuda()

        # 거리 계산
        # distances = self.calculate_distances(global_logit, outputsU)
        similarites = self.calculate_similarities(global_logit, outputsU)

        # min_distances, min_indices = distances.min(dim=1)
        max_similarities, max_indices = similarites.max(dim=1)
        # min_indices = distances.min(dim=1)[1]
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
        crossloss = crossentropyloss(outputs_combined, labels_combined)
        return crossloss

    def calculate_distances(self, logits, outputs):
        distances = []
        for logit in logits:
            distance = torch.norm(logit - outputs, dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)

    def calculate_similarities(self, logits, outputs):
        similarities = []
        for logit in logits:
            similarity = F.cosine_similarity(logit, outputs, dim=1)
            similarities.append(similarity)
        return torch.stack(similarities, dim=1)


class PLoss_my4(nn.Module):
    # cosine 유사도 계산 with sim1 - sim2 + 상위 10
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    def forward(self, outputs, labels, global_logit):
        outputs = outputs.cuda().float()
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).cuda()
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask).cuda()

        # 거리 계산
        # distances = self.calculate_distances(global_logit, outputsU)
        similarities = self.calculate_similarities(global_logit, outputsU)

        # 각 output에 대한 최고 유사도와 그 다음 유사도 찾기 및 차이 계산
        top_two_similarities, top_two_indices = similarities.topk(2, dim=1, largest=True)
        similarity_differences = top_two_similarities[:, 0] - top_two_similarities[:, 1]

        # 상위 10% 유사도 차이에 해당하는 인덱스 선택
        num_selected = int(len(similarity_differences) * 0.1)  # 10%
        top_similarity_diff_indices = similarity_differences.topk(num_selected, largest=True).indices

        # 상위 10%에 해당하는 outputs와 labels만 선택
        selected_outputs = torch.index_select(outputsU, 0, top_similarity_diff_indices)
        selected_labels = torch.index_select(top_two_indices[:, 0], 0, top_similarity_diff_indices)

        # 선택된 outputs와 labels를 기존의 outputsP와 labelsP에 추가
        labels_combined = torch.cat([labelsP, selected_labels], dim=0)
        outputs_combined = torch.cat([outputsP, selected_outputs], dim=0)

        crossentropyloss = nn.CrossEntropyLoss().cuda()
        crossloss = crossentropyloss(outputs_combined, labels_combined)

        return crossloss

    def calculate_distances(self, logits, outputs):
        distances = []
        for logit in logits:
            distance = torch.norm(logit - outputs, dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)

    def calculate_similarities(self, logits, outputs):
        similarities = []
        for logit in logits:
            similarity = F.cosine_similarity(logit, outputs, dim=1)
            similarities.append(similarity)
        return torch.stack(similarities, dim=1)


class PLoss_my5(nn.Module):  # cosine 유사도 계산 sim2 / sim1 + 상위 10%
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    def forward(self, outputs, labels, global_logit):
        outputs = outputs.cuda().float()
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).cuda()
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask).cuda()

        # 거리 계산
        # distances = self.calculate_distances(global_logit, outputsU)
        similarities = self.calculate_similarities(global_logit, outputsU)

        # 각 output에 대한 최고 유사도와 그 다음 유사도 찾기 및 차이 계산
        top_two_similarities, top_two_indices = similarities.topk(2, dim=1, largest=True)
        similarity_differences = top_two_similarities[:, 1] / top_two_similarities[:, 0]

        # 상위 10% 유사도 차이에 해당하는 인덱스 선택
        num_selected = int(len(similarity_differences) * 0.1)  # 10%
        top_similarity_diff_indices = similarity_differences.topk(num_selected, largest=True).indices

        # 상위 10%에 해당하는 outputs와 labels만 선택
        selected_outputs = torch.index_select(outputsU, 0, top_similarity_diff_indices)
        selected_labels = torch.index_select(top_two_indices[:, 0], 0, top_similarity_diff_indices)

        # 선택된 outputs와 labels를 기존의 outputsP와 labelsP에 추가
        labels_combined = torch.cat([labelsP, selected_labels], dim=0)
        outputs_combined = torch.cat([outputsP, selected_outputs], dim=0)

        crossentropyloss = nn.CrossEntropyLoss().cuda()
        crossloss = crossentropyloss(outputs_combined, labels_combined)

        return crossloss

    def calculate_distances(self, logits, outputs):
        distances = []
        for logit in logits:
            distance = torch.norm(logit - outputs, dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)

    def calculate_similarities(self, logits, outputs):
        similarities = []
        for logit in logits:
            similarity = F.cosine_similarity(logit, outputs, dim=1)
            similarities.append(similarity)
        return torch.stack(similarities, dim=1)


class MPULoss_V2(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW

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

        pu3 = (-torch.log(1 - outputsU_Soft + eps) * new_P_indexlist).sum() / \
              max(1, outputsU.size(0)) / len(indexlist)
        PULoss += pu3
        if self.numClass > len(indexlist):
            pu1 = (-torch.log(1 - outputsP_Soft + eps) * new_P_indexlist).sum() * \
                  priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass - len(indexlist))
            PULoss += pu1

        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass * 2).cuda().scatter_(1, torch.unsqueeze(labelsP, 1),
                                                                                         1)[:, :self.numClass]
        log_res = -torch.log(1 - outputsP_Soft * label_onehot_P + eps)
        pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
        PULoss += pu2

        crossentropyloss = nn.CrossEntropyLoss()
        crossloss = crossentropyloss(outputsP, labelsP)

        objective = PULoss * self.puW + crossloss

        return objective, PULoss * self.puW, crossloss
