#mkidr ./data
#wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P ./data
#unzip ./data/ml-100k.zip -d ./data

python3 split.py --input ./data/ml-10M100K/ratings.dat --sep ::

python3 ./encoderder/encoderder.py -c ./100k.json

python3 train_predict.py --train ./exp/ml.train --test ./exp/ml.test.all_pair --output ./result/output.txt

python3 gen_user_pred.py --score_file ./result/output.txt --truth_file ./exp/ml.test.all_pair

python3 eval.py --predict ./result/user_pred_dict.pkl --truth ./exp/ml.test

