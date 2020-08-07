import os
from tokenizers import CharBPETokenizer

zulu_corpus = ['data/autshumato/isizulu.txt', 'data/nchlt/isizulu.txt']

tokenizer = CharBPETokenizer()
tokenizer.train(zulu_corpus)

if not os.path.exists('tokenizers'):
    os.makedirs('tokenizers')

tokenizer.save('tokenizers/isizulu.json')

# Example usage
encoded = tokenizer.encode("Molweni ndisaphila nkosi.")
print(encoded.tokens)

# TODO: make it so that there isn't a word separator </w> before full stop token
