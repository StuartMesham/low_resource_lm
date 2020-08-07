import os
from tokenizers import CharBPETokenizer, Tokenizer

# https://github.com/huggingface/tokenizers/tree/master/bindings/python

zulu_corpus = ['data/autshumato/isizulu.txt', 'data/nchlt/isizulu.txt']

tokenizer = CharBPETokenizer()
tokenizer.train(zulu_corpus)

if not os.path.exists('tokenizers'):
    os.makedirs('tokenizers')

tokenizer.save('tokenizers/isizulu.json')

# Example usage
# tokenizer = Tokenizer.from_file("tokenizers/isizulu.json")
encoded = tokenizer.encode("Molweni ndisaphila nkosi.")
print(encoded.tokens)

# TODO: make it so that there isn't a word separator </w> before full stop token
