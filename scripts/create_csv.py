import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Creates a CSV of the results from the training experiment log file")
parser.add_argument('--log_file', default='logs/experiment_logs.txt', help='Input log file to convert to CSV.')
parser.add_argument('--output_file', default='logs/results.csv', help='Path to output CSV file.')

args = parser.parse_args()

dicts = []

with open(args.log_file, 'r') as f:
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
df.to_csv(args.output_file, index=False)
