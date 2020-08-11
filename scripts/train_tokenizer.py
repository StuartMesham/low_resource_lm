import os
from tokenizers import CharBPETokenizer

# https://github.com/huggingface/tokenizers/tree/master/bindings/python

zulu_corpus = ['data/autshumato/isizulu.txt', 'data/nchlt/isizulu.txt']

tokenizer = CharBPETokenizer()
tokenizer.train(zulu_corpus)

if not os.path.exists('tokenizers/isizulu'):
    os.makedirs('tokenizers/isizulu')

tokenizer.save('tokenizers/isizulu')

# Example usage
tokenizer = CharBPETokenizer(merges_file='tokenizers/isizulu/merges.txt', vocab_file='tokenizers/isizulu/vocab.json')
encoded = tokenizer.encode("Molweni ndisaphila nkosi.")
print(encoded.tokens)

# TODO: make it so that there isn't a word separator </w> before full stop token
