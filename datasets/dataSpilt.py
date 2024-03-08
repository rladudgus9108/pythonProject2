import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from options import opt
import matplotlib.pyplot as plt


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# -------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
# -------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        # if self.inputs
        return self.inputs.shape[0]


def get_MNIST():  ##################################### 여기서 변경
    dataset_train = datasets.MNIST(root=opt.data_root, train=True, download=True,
                                   transform=get_default_data_transforms(opt.dataset, verbose=False)[0])
    dataset_test = datasets.MNIST(root=opt.data_root, train=False, download=True,
                                  transform=get_default_data_transforms(opt.dataset, verbose=False)[1])
    # print(dataset_train.train_data.numpy())
    # print("--------")
    # print(dataset_train.train_labels.numpy())
    # print("--------")
    # print(dataset_test.test_data.numpy())
    # print("--------")
    # print(dataset_test.test_labels.numpy())
    # print("--------")
    # print(dataset_train)

    return dataset_train.train_data.numpy(), dataset_train.train_labels.numpy(), dataset_test.test_data.numpy(), dataset_test.test_labels.numpy()


def get_FMNIST():  ##################################### 여기서 변경
    dataset_train = datasets.FashionMNIST(root=opt.data_root, train=True, download=True,
                                          transform=get_default_data_transforms(opt.dataset, verbose=False)[0])
    dataset_test = datasets.FashionMNIST(root=opt.data_root, train=False, download=True,
                                         transform=get_default_data_transforms(opt.dataset, verbose=False)[1])

    return dataset_train.train_data.numpy(), dataset_train.train_labels.numpy(), dataset_test.test_data.numpy(), dataset_test.test_labels.numpy()


def get_CIFAR10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = datasets.CIFAR10(root=opt.data_root, train=True, download=True)  # default
    data_test = datasets.CIFAR10(root=opt.data_root, train=False, download=True)

    # 아래 있는 코드로 해도 동일한 결과 나옴 transform을 안되어 있어서 한번 해봄
    # data_train = datasets.CIFAR10(root=opt.data_root, train=True, download=True,
    #                               transform=get_default_data_transforms(opt.dataset, verbose=False)[0])
    # data_test = datasets.CIFAR10(root=opt.data_root, train=False, download=True,
    #                              transform=get_default_data_transforms(opt.dataset, verbose=False)[1])

    # 원래 주석
    # x_train, y_train = data_train.train_data.transpose((0, 3, 1, 2)), np.array(data_train.train_labels)
    # x_test, y_test = data_test.test_data.transpose((0, 3, 1, 2)), np.array(data_test.test_labels)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_SVHN():
    data_train = datasets.SVHN(root=opt.data_root, split='train', download=True,
                               transform=get_default_data_transforms(opt.dataset, verbose=False)[0])
    data_test = datasets.SVHN(root=opt.data_root, split='test', download=True,
                              transform=get_default_data_transforms(opt.dataset, verbose=False)[1])

    x_train, y_train = data_train.data, np.array(data_train.labels)
    x_test, y_test = data_test.data, np.array(data_test.labels)

    return x_train, y_train, x_test, y_test


# 수정함, 다른 dataset에 맞게끔 transform을 바꿈
def get_default_data_transforms(name, train=True, verbose=True):
    transforms_train = {
        # 'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'MNIST': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'FMNIST': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            # FashionMNIST
            # 0.2860405969887955     -> 0.2860
            # 0.3530242445149223     -> 0.3530
        ]),
        'SVHN': transforms.Compose([
            transforms.ToPILImage(),  # 이 부분을 제외하고는 동일함
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                 std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4), # 일반화 성능 향상을 위하여 실행함
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),

    }
    transforms_eval = {
        'MNIST': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'FMNIST': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]),
        'SVHN': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                 std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return transforms_train[name], transforms_eval[name]


def relabel_K(dataset_train, unlabel_dict):
    count = 0
    for index, label in enumerate(dataset_train.labels):
        if count < len(unlabel_dict) and index == unlabel_dict[count]:
            dataset_train.labels[index] += opt.num_classes  # 여기에서 + 10을 해줌, unlabel 데이터로 만들기 위해서
            count += 1  # 애초에 보면 dataset_train에 접근하여서 label 자체를 변경하는 구조 원래 1이 였다면 11으로 바꿈
    return dataset_train


def puSpilt_index(dataset, indexlist, samplesize):
    # label 과 unlabel data를 나누는 함수
    labels = dataset.labels.numpy()

    labeled_size = 0
    for i in indexlist:
        labeled_size += int(samplesize[i] * opt.positiveRate)
    unlabeled_size = len(labels) - labeled_size

    # l_shard = [i for i in range(int(singleClass * pos_rate))]
    labeled = np.array([], dtype='int64')
    unlabeled = np.array([], dtype='int64')
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 수직으로 행렬 결합(2,:)와 같은 꼴이됨
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 이 코드가 이해가 되지 않음. 당연히 정렬되어 있기 때문에 0 1 2 3 으로 가야하는데
    # 얘는 왜 0 803 802 ,,, 이런식으로 가는건지 일단 seed 문제는 아닌듯함
    idxs = idxs_labels[0, :]
    priorlist = []  # unlabel data의 차지하는 비중

    # divide to unlabeled
    bias = 0
    for i in range(opt.num_classes):  # ************ 여기가 문제 발생 구간 *******************
        if samplesize[i] != 0:
            if i in indexlist and samplesize[i] >= 40:
                labeled = np.concatenate(
                    (labeled, idxs[bias: int(bias + opt.positiveRate * samplesize[i])]), axis=0)
                bias += int(opt.positiveRate * samplesize[i])
                unlabeled = np.concatenate(
                    (unlabeled, idxs[bias: int(bias + (1 - opt.positiveRate) * samplesize[i])]), axis=0)
                bias += int((1 - opt.positiveRate) * samplesize[i])
                priorlist.append(samplesize[i] * (1 - opt.positiveRate) / unlabeled_size)
            else:
                unlabeled = np.concatenate((unlabeled, idxs[bias: bias + samplesize[i]]), axis=0)
                bias += samplesize[i]
                priorlist.append(samplesize[i] / unlabeled_size)
        else:
            priorlist.append(0.0)

    return labeled, unlabeled, priorlist


def puSpilt_index_my(dataset, indexlist, samplesize):
    # label 과 unlabel data를 나누는 함수
    labels = dataset.labels.numpy()

    labeled_size = 0
    for i in indexlist:
        labeled_size += int(samplesize[i] * opt.positiveRate)
    unlabeled_size = len(labels) - labeled_size

    # l_shard = [i for i in range(int(singleClass * pos_rate))]
    labeled = np.array([], dtype='int64')
    unlabeled = np.array([], dtype='int64')
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 수직으로 행렬 결합(2,:)와 같은 꼴이됨
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 이 코드가 이해가 되지 않음. 당연히 정렬되어 있기 때문에 0 1 2 3 으로 가야하는데
    # 얘는 왜 0 803 802 ,,, 이런식으로 가는건지 일단 seed 문제는 아닌듯함
    idxs = idxs_labels[0, :]
    priorlist = []  # unlabel data의 차지하는 비중

    # divide to unlabeled
    bias = 0
    for i in range(opt.num_classes):  # ************ 여기가 문제 발생 구간 *******************
        if samplesize[i] != 0:
            if i in indexlist and samplesize[i] >= 40:
                labeled = np.concatenate(
                    (labeled, idxs[bias: int(bias + opt.positiveRate * samplesize[i])]), axis=0)
                bias += int(opt.positiveRate * samplesize[i])
                unlabeled = np.concatenate(
                    (unlabeled, idxs[bias: bias + samplesize[i] - int(opt.positiveRate * samplesize[i])]),
                    axis=0)  # unlabel에서 문제가 있었음
                bias += (samplesize[i] - int(opt.positiveRate * samplesize[i]))
                priorlist.append(samplesize[i] * (1 - opt.positiveRate) / unlabeled_size)
            else:
                unlabeled = np.concatenate((unlabeled, idxs[bias: bias + samplesize[i]]), axis=0)
                bias += samplesize[i]
                priorlist.append(samplesize[i] / unlabeled_size)
        else:
            priorlist.append(0.0)

    return labeled, unlabeled, priorlist


# -------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
# -------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
    different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    '''
    n_data = len(data)
    n_labels = np.max(labels) + 1

    data_per_client = [n_data // n_clients] * n_clients
    data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        # 라벨 별로 data가 몇 번째에 있는지 인덱스 번호 저장
        # 여기서 알아둬야 할꺼는 라벨별로 6000개씩 저장된게 아님 라벨 0은 5,923개 1은 6742개 이런식으로 저장됨
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)
        # 각 라벨 별로 인덱스 번호 저장한 것의 위치를 섞음
        # 원래는 라벨 1의 인덱스 위치가 [1 2 3] 이였다면 shuffle을 통하여 [2 3 1] 이런식으로 섞임

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = max(c, 0)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:  # verbose : 장황한, 상세한
        print_split(clients_split)
    return clients_split


def get_data_loaders(verbose=True):  # verbose : 상세한, 장황한
    x_train, y_train, x_test, y_test = globals()['get_' + opt.dataset]()
    # dataset_train, dataset_test = globals()['get_' + opt.dataset]()

    transforms_train, transforms_eval = get_default_data_transforms(opt.dataset, verbose=False)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval),
                                              batch_size=opt.pu_batchsize, shuffle=True)

    split = split_image_data(x_train, y_train, n_clients=opt.num_clients,
                             classes_per_client=opt.classes_per_client,
                             verbose=verbose)
    # 여기에서 각 client 별 데이터가 어떻식으로 들어가는지 정해짐
    # Client 0 : [1200 1200 1200 1200 1200 0 0 0 0 0] 이 부분에서 결정됨

    train_dataset = []
    priorlist = []
    indexlist = []  # 防止返回值出错 -> Prevention of return value errors

    count = 0
    randomIndex_num = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # default
    # randomIndex_num = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # randomIndex_num = [4, 4, 3, 3, 2, 2, 1, 1, 1, 1]
    # randomIndex_num = [2, 2, 2, 2, 2]
    # randomIndex_num = [5,5]

    for i, (x, y) in enumerate(split):
        indexList = []
        dataset = CustomImageDataset(x, y, transforms_train)
        selectcount = [0 * 1 for i in range(opt.num_classes)]

        # 计算每一类的样本量 -> Calculate the sample size for each category
        # 각 client 별로 라벨 당 데이터가 몇개씩 들어가 있는지 저장하는게 samplesize
        samplesize = [0 * 1 for i in range(opt.num_classes)]
        for l in dataset.labels:
            samplesize[l] += 1
        if opt.P_Index_accordance:  # indexlist长度一致 -> Index list length is consistent
            # default : opt.P_Index_accordance is False, P_Index_accordance가 나타내는게 뭔지 정확하게 파악 못함
            for j in range(opt.randomIndex_num):
                k = 0
                while True:
                    index = (count + j + k) % opt.num_classes
                    if (i == (opt.num_clients - 1) or samplesize[index] > 40) and selectcount[
                        index] < opt.randomIndex_num \
                            and (sum(m == 0 for m in selectcount) > (
                            opt.num_classes - opt.classes_per_client) and index not in indexList):
                        indexList.append(index)
                        selectcount[index] += 1
                        break
                    elif k > opt.num_classes:
                        break
                    k += 1
        else:
            for j in range(randomIndex_num[i]):  # 여기에서 각 client에 들어갈 라벨을 정해줌 (변수명 index를 사용하여)
                k = 0
                while True:
                    index = (count + j + k) % opt.num_classes
                    if samplesize[index] > 40 and selectcount[index] < sum(
                            randomIndex_num) / opt.num_classes and index not in indexList:
                        indexList.append(index)
                        selectcount[index] += 1
                        break
                    elif k > opt.num_classes:
                        break
                    k += 1
        label_dict, unlabel_dict, priorList = puSpilt_index_my(dataset, indexList, samplesize)
        priorlist.append(priorList)
        # convert to onehot for torch
        li = [0] * opt.num_classes  # li : label된 label이 무엇인지
        for i in indexList:
            li[i] = 1
        indexlist.append(li)

        unlabel_dict = np.sort(unlabel_dict)  # dict序列排序 -> dict sequence ordering
        if 'SL' not in opt.method:
            dataset = relabel_K(dataset, unlabel_dict)  # 将挑出的unlabeled数据标签全部改为classnum-1
            # -> Change all unlabeled data labels to classnum-1
        train_dataset.append(dataset)
        count += len(indexList)
    print("IndexList")
    print(indexlist)  # indexlist가 무엇인지 알아야 할듯

    client_loaders = [torch.utils.data.DataLoader(
        data, batch_size=opt.pu_batchsize, num_workers=0, shuffle=True) for data in train_dataset]

    # 여기로 파악되는데 num_workers = 16에서 0으로 수정했다고 하셨음

    stats = [x.shape[0] for x, y in split]

    indexlist = torch.Tensor(indexlist).cuda()
    priorlist = torch.Tensor(priorlist).cuda()

    return client_loaders, stats, test_loader, indexlist, priorlist
    # torch.Tensor(indexlist).cuda(), torch.Tensor(priorlist).cuda()


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))
