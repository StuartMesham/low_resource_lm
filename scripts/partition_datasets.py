import argparse

parser = argparse.ArgumentParser(description="Separates the datasets into test/train/validate sets")
parser.add_argument('--test_split', default=10, help='Percentage of data to be used for testing')
parser.add_argument('--train_split', default=80, help='Percentage of data to be used for training')
parser.add_argument('--valid_split', default=10, help='Percentage of data to be used for validation')
parser.add_argument('--autshumato_dir', default='data/autshumato/',
                    help='directory where autshumato files can be found')
parser.add_argument('--isolezwe_dir', default='data/isolezwe/', help='directory where isolezwe files can be found')
parser.add_argument('--nchlt_dir', default='data/nchlt/', help='directory where nchlt files can be found')
parser.add_argument('--awd_lstm', default=False, help='Prepends a space char to all lines for awd-lstm implementation')

args = parser.parse_args()
assert (args.test_split + args.train_split + args.valid_split == 100), "Dataset splits must add to 100"

isolezwe_files = ['isizulu.txt']
autshumato_files = ['isizulu.txt', 'sepedi.txt']
nchlt_files = ['isizulu.txt', 'sepedi.txt']
datasets = [[args.autshumato_dir, autshumato_files],
            [args.isolezwe_dir, isolezwe_files],
            [args.nchlt_dir, nchlt_files]]

for dataset in datasets:
    for file in dataset[1]:
        with open(dataset[0] + file[:-4] + '/' + file, 'r', encoding='utf-8') as inf:
            corpus = inf.read()
            if args.awd_lstm:
                corpus = ' ' + corpus.replace('\n', '\n ')

            with open(dataset[0] + file[:-4] + '/test.txt', 'w', encoding='utf-8') as f:
                f.write(corpus[:int(len(corpus) * args.test_split / 100)])

            with open(dataset[0] + file[:-4] + '/train.txt', 'w', encoding='utf-8') as f:
                f.write(corpus[int(len(corpus) * args.test_split / 100):int(len(corpus) * args.train_split / 100) + int(
                    len(corpus) * args.test_split / 100)])

            with open(dataset[0] + file[:-4] + '/valid.txt', 'w', encoding='utf-8') as f:
                f.write(corpus[int(len(corpus) * args.train_split / 100) + int(len(corpus) * args.test_split / 100):])
