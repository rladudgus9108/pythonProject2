# client 100 contrast
python main.py  --dataset 'CIFAR10' --pu_batchsize 1024 --positiveRate 0.3 --pu_lr 0.01 --local_epochs 30 --P_Index_accordance --randomIndex_num 10 --num_clients 100 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.05