import pickle
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--predict", help="the predict pickle file")
parser.add_argument("--truth", help="the truth file")
args = parser.parse_args()



def find_dcg(element_list):
    """
    Discounted Cumulative Gain (DCG)
    The definition of DCG can be found in this paper:
        Azzah Al-Maskari, Mark Sanderson, and Paul Clough. 2007.
        "The relationship between IR effectiveness measures and user satisfaction."
    Parameters:
        element_list - a list of ranks Ex: [5,4,2,2,1]
    Returns:
        score
    """
    score = 0.0
    for order, rank in enumerate(element_list):
        score += float(rank)/math.log((order+2))
    return score


def ndcg(reference, hypothesis):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG:
        nDCG = DCG(hypothesis)/DCG(reference)
    Parameters:
        reference   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis  - a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        ndcg_score  - normalized score
    """

    return find_dcg(hypothesis)/find_dcg(reference)
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

def precision(actual: List[list], predicted: List[list]) -> int:
    """
    Computes the precision of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        precision: int
    """
    def calc_precision(actual,predicted):
        prec = [value for value in predicted if value in actual]
        prec = np.round(float(len(prec)) / float(len(predicted)), 4)
        return prec

    precision = np.mean(list(map(calc_precision, predicted, actual)))
    return precision


def recall(actual: List[list], predicted: List[list]) -> int:
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

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



print(f"map@10 : {mapk(actual_list, predict_list, 10)}")
print(f"ap@10 : {precision(actual_list, predict_list, 10)}")
print(f"recall@10 : {recall(actual_list, predict_list, 10)}")
print(f"ndcg@10 : {ndcg(actual_list, predict_list, 10)}")

