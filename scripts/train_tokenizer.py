import os
from tokenizers import CharBPETokenizer

# https://github.com/huggingface/tokenizers/tree/master/bindings/python

zulu_corpus = ['data/autshumato/isizulu.txt', 'data/nchlt/isizulu.txt']

tokenizer = CharBPETokenizer()
tokenizer.train(zulu_corpus)

if not os.path.exists('tokenizers/isizulu'):
    os.makedirs('tokenizers/isizulu')

tokenizer.save('tokenizers/isizulu.json')

# Example usage
# tokenizer = Tokenizer.from_file('tokenizers/isizulu.json')
encoded = tokenizer.encode("Molweni ndisaphila nkosi.")
print(encoded.tokens)
