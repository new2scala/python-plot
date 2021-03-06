def read_conll(fname):

    r = []
    with open(fname) as f:
        lines = f.readlines()
        trimmed = [l.strip() for l in lines]
        curr_lst = []
        for tl in trimmed:
            words = tl.split()
            if (len(words) > 0):
                curr_lst.append(tuple(words))
            else:
                r.append(curr_lst)
                curr_lst = []

    return r

lst = read_conll('/media/sf_work/aff-data/usa-train-converted.txt')

import pickle
pickle.dump(lst, open("usa_train.pkl", "wb"))
