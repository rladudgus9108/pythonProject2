# client 100 contrast
CUDA_VISIBLE_DEVICES=1 python main.py  --dataset 'CIFAR10' --pu_batchsize 1024 --positiveRate 0.33 --pu_lr 0.001 --local_epochs 10 --P_Index_accordance --randomIndex_num 10 --num_clients 100 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.05 > nohup_v2.out 2>&2 &
