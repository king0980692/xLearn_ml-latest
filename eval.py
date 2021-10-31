import pickle
import numpy as np

def apk(actual, predicted, k=10):

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0


    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


with open("./result/user_pred_dict.pkl",'rb') as p:
    tmp_user_pred =  pickle.load(p)

user_pred = {}
for user in tmp_user_pred:
    for rec in tmp_user_pred[user]:
        item, rating = rec
        if user not in user_pred:
            user_pred[user] = []
        user_pred[user].append(item)

user_actual = {}
with open("./exp/ml.test",'r') as f:
    for line in f.readlines():
        rating, uid, iid = line.rstrip().split()
        uid = uid.split(":")[0]
        iid = iid.split(":")[0]
        if uid not in user_actual:
            user_actual[uid] = []

        user_actual[uid].append(iid)

actual_list = []
predict_list = []
for user in user_actual:
    if user in user_pred:
        actual_list.append(user_actual[user])
        predict_list.append(user_pred[user])



print(f"map@10 : {mapk(actual_list, predict_list, 10)}")



