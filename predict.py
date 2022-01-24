import sys
import numpy as np
import itertools

def read_wei(file):

    bias = -1
    ori_wei = [0]
    first_ord_wei = []
    second_ord_wei = []
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

    
    norm = 0.5

    pred += (first_ord[uid])*np.sqrt(norm)
    pred += (first_ord[iid])*np.sqrt(norm)


    #print("first : ",pred)
    pred += np.dot(second_ord[uid]*norm,second_ord[iid]*norm)

    #print("second : ",np.dot(second_ord[uid],second_ord[iid]))

    return pred


def read_test(test_file,bias, first_ord, second_ord):

    with open(test_file, 'r') as f:

        out_list = []
        for line in f:
            pred = bias
            line = line.rstrip().split()
            u_id, u_val = line[0].split(":")
            i_id, i_val = line[1].split(":")

            u_id = int(u_id)
            u_val = float(u_val)
            i_id = int(i_id)
            i_val = float(i_val)
            
            norm = 1 / (u_val*u_val + i_val*i_val)


            pred += first_ord[u_id]*u_val*np.sqrt(norm)
            pred += first_ord[i_id]*i_val*np.sqrt(norm)

            pred += np.dot(second_ord[u_id],second_ord[i_id])


if __name__ == '__main__':
    wei_file, test_file ,out_file = sys.argv[1:]

    bias, first_ord_wei, second_ord_wei = read_wei(wei_file)
    #print(emb_score(1,944,bias,first_ord_wei,second_ord_wei))

    #read_test(test_file,bias, first_ord_wei, second_ord_wei)

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




