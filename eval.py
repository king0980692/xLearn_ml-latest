import pickle
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--predict", help="the predict pickle file")
parser.add_argument("--truth", help="the truth file")
args = parser.parse_args()



def mapk(actual, predicted, k=10):
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

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def recall(actual, predicted, k):
    def calc_recall(predicted, actual):
        score = 0.0
        num_hits = 0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1

        if not num_hits:
            return 0.0

        for i in range(num_hits,1):
            score += 1.0/num_hits

        return score

    print(list(map(calc_recall, predicted, actual)))
    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall

with open(args.predict,'rb') as p:
    tmp_user_pred =  pickle.load(p)

user_pred = {}
for user in tmp_user_pred:
    for rec in tmp_user_pred[user]:
        item, rating = rec
        if user not in user_pred:
            user_pred[user] = []
        user_pred[user].append(item)


user_actual = {}
with open(args.truth,'r') as f:
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


print(f"   map@10 : {mapk(actual_list, predict_list, 10)}")
#print(f"recall@10 : {recall(actual_list, predict_list, 10)}")
#print(f"ndcg@10 : {ndcg(actual_list, predict_list)}")

