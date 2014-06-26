from collections import OrderedDict

def groupby(seq, key):
    tmp = OrderedDict()
    for ss in seq:
        tmp.setdefault(key(ss), []).append(ss)
    return tmp

