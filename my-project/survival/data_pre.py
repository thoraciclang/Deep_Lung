import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort Training Data for Accurate Likelihood
    # sort array using pandas.DataFrame(According to DESC 't' and ASC 'e')
    df1 = pd.DataFrame({'t': t, 'e': e})
    df1.sort_values(['t', 'e'], ascending=[False, True], inplace=True)
    sort_idx = list(df1.index)
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, {'e': e, 't': t}

def parse_data(x, label):
    # sort data by t
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    max_time = np.max([int(tt) for tt in label['t']])
    failures = {}
    atrisk = {}
    n, cnt = 0, 0
    t_vector = []
    for i in range(len(e)):
        if e[i]:
            temp = [0 if i < int(t[i]) else 1 for i in range(max_time)]
            t_vector.append(temp)
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                # ties occured
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
        else:
            temp = [0 for i in range(max_time)]
            t_vector.append(temp)

    # when ties occured frequently
    if cnt >= n / 2:
        ties = 'efron'
    elif cnt > 0:
        ties = 'breslow'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties
