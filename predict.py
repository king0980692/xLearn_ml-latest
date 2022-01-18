import sys
import numpy as np
import itertools

def read_wei(file):

    bias = -1
    first_ord_wei = [0]
    second_ord_wei = [0]
    with open(file, 'r') as f:
        bias = float(f.readline().rstrip().split()[-1])

        for line in f:
            line = line.rstrip().split()
            if line[0].startswith('i_'):
                first_ord_wei.append(float(line[1]))
            elif line[0].startswith('v_'):
                second_ord_wei.append(np.array([float(l) for l in line[1:]]))


    #print(len(first_ord_wei))
    #print(len(second_ord_wei))
    return bias, first_ord_wei, second_ord_wei

def emb_score(uid, iid, bias, first_ord, second_ord):

    pred = bias
    pred += first_ord[uid]
    pred += first_ord[iid]

    pred += np.dot(second_ord[uid],second_ord[iid])

    return pred



if __name__ == '__main__':
    wei_file ,out_file = sys.argv[1:]

    bias, first_ord_wei, second_ord_wei = read_wei(wei_file)

    with  open(out_file,'w') as o:
        cnt = 0
        out_list = []
        for uid in range(1,944):
            for iid in range(944,2624):
                pred = emb_score(uid, iid, bias, first_ord_wei, second_ord_wei)

                out_list.append(str(pred)+"\n")
        o.writelines(out_list)
        #for line in f:
        #    line = line.rstrip().split()[1:]
        #    pred = bias

        #    idx_list = []
        #    for l in line:
        #        feat_idx = int(l.split(':')[0])
        #        idx_list.append(feat_idx)
        #        #print(first_ord_wei[feat_idx]*)
        #        #print(feat_idx)
        #        pred += first_ord_wei[feat_idx]

        #    for x, y in itertools.combinations(idx_list,2):
        #        pred += np.dot(second_ord_wei[x],second_ord_wei[y])
        #    #print(pred)
        #    out_list.append(str(pred)+"\n")
        #    cnt += 1




