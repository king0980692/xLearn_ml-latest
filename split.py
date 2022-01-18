import json
from datetime import datetime
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="the based dataset which will be splited")
parser.add_argument("--sep", help="the seperator")
parser.add_argument("--header",default=False ,help="header or not")
parser.add_argument("--timecut", help="the timecut spliter")
args = parser.parse_args()

train_edges, test_edges = [], []
train_all_edges, test_all_edges = [], []

#time_cut = 1527811200
time_cut = int(args.timecut)
#time_cut = 980095522
with open(args.input, 'r') as f:
    if args.header:
        next(f)
    for line in tqdm(f):
        uid, iid, rate, time = line.rstrip('\n').split(args.sep)

        if int(time) < time_cut:
            train_all_edges.append(f"u{uid}\t{iid}\t{rate}")
        else:
            test_all_edges.append(f"u{uid}\t{iid}\t{rate}")

with open('./exp/ml.train', 'w') as f:
    f.write('\n'.join(train_all_edges))
    f.write('\n')
with open('./exp/ml.test', 'w') as f:
    f.write('\n'.join(test_all_edges))
    f.write('\n')

