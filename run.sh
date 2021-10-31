mkidr ./data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P ./data
unzip ./data/ml-100k.zip -d ./data

python3 ./encoderder/encoderder.py -c ./test.json

python3 train_predict.py

poetry run python3 gen_user_pred.py --score_file ./result/output.txt --truth_file ./data/ua.test

python3 eval.py
