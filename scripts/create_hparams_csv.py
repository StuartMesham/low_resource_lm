import pandas as pd

dicts = []

with open('hparam_test.txt', 'r') as f:
    for line in f:
        dicts.append(eval(line))


def flatten_dict(d):
    temp_d = {}
    for k, v in d.items():
        if type(v) == dict:
            temp_d.update(flatten_dict(v))
        else:
            temp_d[k] = v

    return temp_d


dicts = [flatten_dict(d) for d in dicts]

df = pd.DataFrame(dicts)
df.to_csv('hparam_tests.csv', index=False)
