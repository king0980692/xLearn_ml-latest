import json
from datetime import datetime
from tqdm import tqdm

train_edges, test_edges = [], []
train_all_edges, test_all_edges = [], []
#time_cut = 1514736000
time_cut = 1527811200
with open('data/ml-latest/ratings.csv', 'r') as f:
    next(f)
    for line in tqdm(f):
        uid, iid, rate, time = line.rstrip('\n').split(',')
        if int(time) < time_cut:
            train_all_edges.append(f"u{uid}\t{iid}\t{rate}")
            if float(rate) > 3.:
                train_edges.append(f"u{uid}\t{iid}\t{rate}")
        else:
            test_all_edges.append(f"u{uid}\t{iid}\t{rate}")
            if float(rate) > 3.:
                test_edges.append(f"u{uid}\t{iid}\t{rate}")

with open('./exp/ml.train', 'w') as f:
    f.write('\n'.join(train_edges))
    f.write('\n')
with open('./exp/ml.test', 'w') as f:
    f.write('\n'.join(test_edges))
    f.write('\n')
with open('./exp/ml2.train', 'w') as f:
    f.write('\n'.join(train_all_edges))
    f.write('\n')
with open('./exp/ml2.test', 'w') as f:
    f.write('\n'.join(test_all_edges))
    f.write('\n')

