from tqdm import tqdm
import gc
import mmap
import pickle
import argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--score_file", help="the FM prediction score file")
parser.add_argument("--truth_file", help="the truth file")
args = parser.parse_args()

user_pred_dict = {}
last_user = ""

## Reading score file
with open(args.score_file,'r') as o:
    first_flag = True
    idx = 0
    with open(args.truth_file, "r+b") as f:
        map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        #pbar = tqdm(total=total_len)
        for line in iter(map_file.readline, b""):
            score = o.readline().rstrip()
            idx += 1
            #if float(score) < 4:
            #    continue
            #if idx == 10:
            #    break

            line = line.decode()
            user,item = line.rstrip().split(" ")
            user = user.split(':')[0]
            item = item.split(':')[0]
            #print(user,item)

            if first_flag:
                last_user = user
                first_flag = False
            # change user
            if user != last_user:
                user_pred_dict[last_user].sort(key=lambda x:x[1],reverse=True)
                user_pred_dict[last_user] = user_pred_dict[last_user][:10]
                gc.collect()
                last_user = user
                #pbar.update(1)

            if user not in user_pred_dict:
                user_pred_dict[user] = []
            user_pred_dict[user].append((item,score))


# the last user
user_pred_dict[user].sort(key=lambda x:x[1],reverse=True)
user_pred_dict[user] = user_pred_dict[last_user][:10]


with open("./result/user_pred_dict.pkl",'wb') as p:
    pickle.dump(user_pred_dict,p)

