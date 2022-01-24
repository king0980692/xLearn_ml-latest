#mkdir ./data
#wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P ./data
#unzip ./data/ml-100k.zip -d ./data
set -x

#for neg_sp in 0 5 10 20 50 100
for neg_sp in 20
    do
        python3 to_xlearn.py ./data/ml-100k/ua.base ./data/ml-100k/ua.test ./exp/ml.xl.train ./exp/ml.xl.test ./exp/ml.xl.pairs $neg_sp >/dev/null

        python3 train_predict.py --train ./exp/ml.xl.train --test ./exp/ml.xl.pairs --output ./result/output.txt >/dev/null
        #python3 train_predict.py --train ./exp/ml.xl.train --test ./exp/ml.xl.test --output ./result/output.txt

        #~/xlearn/build/xlearn_train ./exp/ml.xl.train -s 4 -k 40 -r 0.25 -b 0.002 -e 50 -p adagrad -t ./exp/model.txt -m ./exp/model.out

        # ~/xlearn/build/xlearn_predict ./exp/ml.xl.pairs ./exp/model.out -o result/output.txt

        python3 predict.py exp/model.txt exp/ml.xl.pairs result/output.txt

        python3 gen_user_pred.py --score_file ./result/output.txt --truth_file ./exp/ml.xl.pairs >/dev/null

        python3 eval.py --predict ./result/user_pred_dict.pkl --truth ./exp/ml.xl.test | grep '10'

    done

exit
