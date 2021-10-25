python3 split.py ./sub_data/sub_ratings.csv

python3 ../encoderder/main.py -c ./ml_encoderer.json

~/xlearn/build/xlearn_train ./train_dir/train.txt -s 4 --disk

~/xlearn/build/xlearn_predict ./test_dir/test.txt ./train_dir/train.txt.model -o ./output.txt --disk

python3 gen_user_pred.py --score_file ./output.txt --truth_file ./exp/test_dir/test.txt
