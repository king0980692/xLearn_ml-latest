#mkidr ./data
#wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -P ./data
#unzip ./data/ml-1m.zip -d ./data

set -x

python3 split.py --input ./data/ml-latest/ratings.csv --sep ::  --timecut 980095522
#python3 split.py --input ./data/ml-1m/ratings.dat --sep ::

for neg_sp in 0 5 10 20 50 100

    do
        python3 to_xlearn.py ./exp/ml.train ./exp/ml.test ./exp/ml.xl.train ./exp/ml.xl.test ./exp/ml.xl.pairs $neg_sp >/dev/null

        python3 train_predict.py --train ./exp/ml.xl.train --test ./exp/ml.xl.pairs --output ./result/output.txt >/dev/null

        python3 gen_user_pred.py --score_file ./result/output.txt --truth_file ./exp/ml.xl.pairs >/dev/null

        python3 eval.py --predict ./result/user_pred_dict.pkl --truth ./exp/ml.xl.test | grep '10'

    done

exit



# --


python3 split.py --input ./data/ml-1m/ratings.dat --sep ::

python3 ./encoderder/encoderder.py -c ./100k.json

python3 train_predict.py --train ./exp/ml.train --test ./exp/ml.test.all_pair --output ./result/output.txt

python3 gen_user_pred.py --score_file ./result/output.txt --truth_file ./exp/ml.test.all_pair

python3 eval.py --predict ./result/user_pred_dict.pkl --truth ./exp/ml.test
