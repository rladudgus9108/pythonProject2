# client 100 contrast
#python main.py  --dataset 'CIFAR10' --positiveRate 0.01 --randomIndex_num 2 --num_clients 100 --communication_rounds 200 --classes_per_client 10 --clientSelect_Rate 0.05
# client 100 contrast
#python main.py  --positiveRate 0.01 --randomIndex_num 2 --num_clients 100 --communication_rounds 200 --classes_per_client 10 --clientSelect_Rate 0.05
# client 50
#python main.py  --positiveRate 0.01 --randomIndex_num 2 --num_clients 100 --communication_rounds 200 --classes_per_client 10 --clientSelect_Rate 0.05
# client 20
python main.py  --positiveRate 0.33 --P_Index_accordance --randomIndex_num 2 --num_clients 10 --communication_rounds 500 --classes_per_client 10 --clientSelect_Rate 0.2