import argparse
import os
parser = argparse.ArgumentParser(description="Separates the datasets into test/train/validate sets")
parser.add_argument('--test_split', default=10, help='Percentage of data to be used for testing')
parser.add_argument('--train_split', default=80, help='Percentage of data to be used for training')
parser.add_argument('--valid_split', default=10, help='Percentage of data to be used for validation')
parser.add_argument('--autshumato_dir', default='data/autshumato/', help='directory where autshumato files can be found')


args = parser.parse_args()
assert (args.test_split + args.train_split + args.valid_split == 100), "Dataset splits must add to 100"

nchlt_files = ['']
isolezwe_files = ['']
# Autshumato
autshumato_files = ['isizulu.txt', 'sepedi.txt']
for file in autshumato_files:
    with open(args.autshumato_dir + file, 'r', encoding='utf-8') as inf:
        corpus = inf.read()

        with open(args.autshumato_dir + file[:-4] + '_test.txt', 'w', encoding='utf-8') as f:
            f.write(corpus[:int(len(corpus)*args.test_split)])

        with open(args.autshumato_dir + file[:-4] + '_train.txt', 'w', encoding='utf-8') as f:
            f.write(corpus[int(len(corpus)*args.test_split):int(len(corpus)*args.train_split)+int(len(corpus)*args.test_split)])

        with open(args.autshumato_dir + file[:-4] + '_valid.txt', 'w', encoding='utf-8') as f:
            f.write(corpus[int(len(corpus)*args.train_split)+int(len(corpus)*args.test_split):])
