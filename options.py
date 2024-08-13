import argparse

parser = argparse.ArgumentParser()

# seed 고정을 위한 코드
parser.add_argument('--seed', type=int, default=4, help='random seed')
# optimizer
parser.add_argument('--pu_lr', type=float, default=0.005, help='learning rate of each client')  # default: 0.01
parser.add_argument('--adjust_lr', action='store_true', default=True,
                    help='adjust lr according to communication rounds')
parser.add_argument('--pu_batchsize', type=int, default=100, help='batchsize of dataloader')  # default : 500
parser.add_argument('--momentum', type=float, default=0.5, help='optimizer param')  # default : 0.9
# dataset
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
# MNIST, FMNIST, CIFAR10, SVHN
# parser.add_argument('--dataset', type=str, default='CIFAR10') MNIST
parser.add_argument('--data_root', type=str, default='./data/',help='data store root')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes used in dataset')
# PU param
parser.add_argument('--pu_weight', type=float, default=1, help='weight of puloss')  # 1
parser.add_argument('--local_epochs', type=int, default=1, help='epoches of each client')  # default : 20

# pu dataloader
parser.add_argument('--randomIndex_num', type=int, default=2, help='rate of positive sample')
parser.add_argument('--P_Index_accordance', action='store_true', help='the same positive class index number')  # default
parser.add_argument('--positiveRate', type=float, default=0.01, help='rate of positive sample')
# 여기를 변경하면서 실험중, CIFAR10에서 labeling이 어떨때 높게 나오는지를 확인하기 위해서
# default : 0.33
# use Fedmatch dataloader
parser.add_argument('--task', type=str, default='SS')  # default : FedPU
parser.add_argument('--useFedmatchDataLoader', action='store_true',
                    help='the same positive class index number') # 아무것도 없으면 False임
parser.add_argument('--method', type=str, default='FedAvg')  # default : FedAvg
# task : FedPU, SL, SS -> SL, SS 이지 않을까 생각중
# method : FedProx, FedAvg, SL -> FedProx, FedAvg, FM 이지 않을까 생각중

# 내가 생각하기에는 task는 SL, SS 만 있고 method는 FedProx, FedAvg, FM만 있고


# FL aggregator
parser.add_argument('--num_clients', type=int, default=10)  # default : 100 인데 수정함
parser.add_argument('--communication_rounds', type=int, default=200)  # default : 5000 인데 수정함
# 0.01일때, round 100만 되도 ACC 62 정도 나오는데, 500가도 ACC 65임 그래서 100으로 설정함
parser.add_argument('--classes_per_client', type=int, default=10)  # default : 5 인데 수정함
parser.add_argument('--clientSelect_Rate', type=float, default=1.0)  # default : 0.5
# FedProx parameters
parser.add_argument('--mu', type=float, default=0.0)
parser.add_argument('--percentage', type=float, default=0.0)

opt, _ = parser.parse_known_args()

FedAVG_model_path = r'C:\Users\DMLAB\PycharmProjects\pythonProject2\local_model'
FedAVG_aggregated_model_path = r'C:\Users\DMLAB\PycharmProjects\pythonProject2\local_model\FedAVG_model.pth'
