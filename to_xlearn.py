import sys
import random


class label_encoder:
    def __init__(self, offset=0, shared=False):
        self.share_idx = shared
        self.idx = offset
        self.data = None

    def _unique(self):
        self.data = set(self.data)

    def _map_to_int(self):
        # if shared_idx flag is opened, those unseen label will be encoded the same id as "null"
        if self.share_idx == True:
            self.data.add('null')

        # make sure every time the encode is unique
        self.data = sorted(self.data)

        self.table = {val: i + self.idx for i, val in enumerate(self.data)}

        self.inverse_table = {val: i for i, val in self.table.items()}
        #return [table[v] for v in values]

    def _encode(self):
        return self._map_to_int()

    def fit(self, data, offset=0):
        if offset != 0:
            self.idx =offset
        self.data = data
        self._unique()
        self._encode()
        return self

    def transform(self, key):
        if self.data is None:
            raise

        if isinstance(key, list):
            output = []
            for k in key:
                if k in self.table:
                    output.append(self.table[k])
                else:
                    if self.share_idx:
                        output.append(self.table['null'])
                    else:
                        raise

            return output
        else:
            if key in self.table:
                return self.table[key]
            else:
                if self.share_idx:
                  return self.table['null']
                else:
                  return 'null'


def split(fname, n_proc):

    def open_with_header_witten(path, idx, header):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        if not has_header:
            return f
        f.write(header)
        return f

    def count_lines(fname):
        '''
        return the same usage with "wc -l [file name]"
        '''
        def _make_gen(reader):
            b = reader(2 ** 16)
            while b:
                yield b
                b = reader(2 ** 16)
        with open(fname, "rb") as f:
            count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
        return count

    def calc_nr_lines_per_thread(fname):
        nr_lines = count_lines(fname)
        if not has_header:
            nr_lines += 1
        return math.ceil(float(nr_lines)/nr_thread)

    #header = open(fname).readline()
    has_header = False

    nr_lines_per_thread = calc_nr_lines_per_thread(fname)

    idx = 0
    f = open_with_header_witten(fname, idx, header)
    for i, line in enumerate(open_with_first_line_skipped(fname, has_header), start=1):
        if i % nr_lines_per_thread == 0:
            f.close()
            idx += 1
            f = open_with_header_witten(fname, idx, header)
        f.write(line)
    f.close()




if __name__ == '__main__':
    _, train_file, test_file, out_file1, out_file2, all_pair_file, neg_sp = sys.argv


    user_le = label_encoder()
    item_le = label_encoder()

    tmp_usr_lst = []
    tmp_itm_lst = []

    with open(train_file, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            for _id, _feat in enumerate(line):
                # user
                if _id == 0:
                    tmp_usr_lst.append(_feat)
                # item
                elif _id == 1:
                    tmp_itm_lst.append(_feat)

    user_le.fit(tmp_usr_lst,offset=1)

    item_le.fit(tmp_itm_lst,offset=len(set(tmp_usr_lst))+1)

    #print(user_le.transform(tmp_usr_lst))
    #print(item_le.transform(tmp_itm_lst))


    '''
        create the train format
    '''

    print("Create Train format")
    output = []
    with open(out_file1, 'w') as o, open(train_file, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            out = []
            for _id, _feat in enumerate(line):
                # user
                if _id == 0:
                    a = user_le.transform(_feat)
                    out.append(f"{a}:1")
                # item
                elif _id == 1:
                    b = item_le.transform(_feat)
                    out.append(f"{b}:1")
                # ratings
                elif _id == 2:
                    out.insert(0,_feat)

            output.append(" ".join(out)+"\n")
            for i in range(int(neg_sp)):
                rand_usr = random.randint(1, len(user_le.table)+1)
                rand_usr = random.randint(len(user_le.table)+2, len(user_le.table)+len(item_le.table)+2)
                neg_out = " ".join(["-1",f"{rand_usr}:1",f"{b}:1"])
                output.append(neg_out+"\n")

        o.writelines(output)

    '''
        create the test format
    '''

    print("Create Test format")
    output = []
    with open(out_file2, 'w') as o, open(test_file, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            out = []
            for _id, _feat in enumerate(line):
                # user
                if _id == 0:
                    a = user_le.transform(_feat)
                    if a == 'null':
                        break
                    out.append(f"{a}:1")
                # item
                elif _id == 1:
                    b = item_le.transform(_feat)
                    if b == 'null':
                        break
                    out.append(f"{b}:1")
                # ratings
                elif _id == 2:
                    out.insert(0,_feat)

            if len(out) == 3:
                output.append(" ".join(out)+"\n")

        o.writelines(output)



    print("Create All-Pair format")
    '''
        create all pair
    '''
    output = []

    iterator = iter(user_le.inverse_table)
    o = open(all_pair_file, 'w')
    cnt = 10000
    while True:
        try:
            uid = next(iterator)
            for iid in item_le.inverse_table:
                out = f"{uid}:1 {iid}:1\n"
                output.append(out)
            cnt -= 1
            if cnt == 0:
                o.writelines(output)
                output = []
                cnt = 10000

        except StopIteration:
            break

    o.writelines(output)
    o.close()
