import os
import argparse
from tokenizers import CharBPETokenizer

# https://github.com/huggingface/tokenizers/tree/master/bindings/python

parser = argparse.ArgumentParser(description='Train Byte-Pair Encoding Tokenizer.')
parser.add_argument('--vocab_size', required=False, default=5000, type=int, help='directory where output files will be saved')
args = parser.parse_args()

zulu_corpus = ['data/autshumato/isizulu.txt', 'data/nchlt/isizulu.txt']

tokenizer = CharBPETokenizer()
tokenizer.train(zulu_corpus, vocab_size=args.vocab_size)

if not os.path.exists('tokenizers/isizulu'):
    os.makedirs('tokenizers/isizulu')

tokenizer.save_model('tokenizers/isizulu')

# Example usage
# tokenizer = CharBPETokenizer(merges_file='tokenizers/isizulu/merges.txt', vocab_file='tokenizers/isizulu/vocab.json')
encoded = tokenizer.encode("Molweni ndisaphila nkosi.")
print(encoded.tokens)
