import os
from tokenizers import CharBPETokenizer

# https://github.com/huggingface/tokenizers/tree/master/bindings/python

zulu_corpus = ['data/autshumato/isizulu.txt', 'data/nchlt/isizulu.txt']

tokenizer = CharBPETokenizer()
tokenizer.train(zulu_corpus)

if not os.path.exists('tokenizers/isizulu'):
    os.makedirs('tokenizers/isizulu')

tokenizer.save_model('tokenizers/isizulu')

# Example usage
# tokenizer = CharBPETokenizer(merges_file='tokenizers/isizulu/merges.txt', vocab_file='tokenizers/isizulu/vocab.json')
encoded = tokenizer.encode("Molweni ndisaphila nkosi.")
print(encoded.tokens)
