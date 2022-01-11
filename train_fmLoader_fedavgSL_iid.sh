# debug Fedmatch dataloader

# iid  FedAVG-SL
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-biid-c10' --method 'FedAVG-SL' --num_clients 100 --local_epochs 200 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05

python main.py   --dataset 'CIFAR10' --method 'FedAVG-SL' --num_clients 1 --local_epochs 200 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05
