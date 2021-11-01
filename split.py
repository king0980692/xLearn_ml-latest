import json
from datetime import datetime
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="the based dataset which will be splited")
parser.add_argument("--sep", help="the seperator")
parser.add_argument("--header",default=False ,help="header or not")
args = parser.parse_args()


time_list = []
with open(args.input, 'r') as f:
    if args.header:
        next(f)
    for line in tqdm(f):
        uid, iid, rate, time = line.rstrip('\n').split(args.sep)
        time_list.append(int(time))

time_list.sort()
cut_index = int(len(time_list)*0.8)

time_cut = time_list[cut_index]

train_edges, test_edges = [], []
train_all_edges, test_all_edges = [], []

with open(args.input, 'r') as f:
    if args.header:
        next(f)
    for line in tqdm(f):
        uid, iid, rate, time = line.rstrip('\n').split(args.sep)

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

