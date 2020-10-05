# Written by Stuart Mesham (MSHSTU001) in full

import re
import unicodedata
from collections import Counter
from statistics import median, stdev

# with help from:
# https://stackoverflow.com/questions/10294032/python-replace-typographical-quotes-dashes-etc-with-their-ascii-counterparts

quote_characters = []
for i in range(0, 0x10ffff):
    char = chr(i)

    try:
        if 'QUOTATION MARK' in unicodedata.name(char):
            quote_characters.append(char)
    except:
        pass


def normalize_quote_characters(str):
    for quote_char in quote_characters:
        str = str.replace(quote_char, '"')

    return str


def clean_sentences(sentences, min_length=1, illegal_substrings=[], lines_to_remove=0):
    # remove first n lines
    sentences = sentences[lines_to_remove:]

    # remove sentences with illegal substrings
    for substring in illegal_substrings:
        sentences = [sentence for sentence in sentences if substring not in sentence]

    # remove numberings (e.g. "1.4.11.1")
    sentences = [re.sub(r'(\d+\.+)+\d*', '', sentence) for sentence in sentences]

    # remove numberings (e.g. "(5) (a) (iiv)")
    sentences = [re.sub(r'\(.|[iv\d]+\)', '', sentence) for sentence in sentences]

    # remove non-alphabet sentence starts and sentences with no alphabet characters
    sentences = [sentence[re.search(r'[a-zA-Z]', sentence).start():] for sentence in sentences if re.search(r'[a-zA-Z]', sentence) is not None]

    # remove extra whitespace
    sentences = [sentence.strip() for sentence in sentences]

    # remove too short sentences
    sentences = [sentence for sentence in sentences if len(sentence) >= min_length]

    # un-caps-lock first words in some sentences
    sentences = [re.sub(r'[A-Z -]+', lambda m: m.group().capitalize(), sentence, count=1) for sentence in sentences]

    # limit to max 3 consecutive dots in a row
    sentences = [re.sub(r'\.\.\.\.*', '...', sentence) for sentence in sentences]

    # remove too frequent or too short sentences
    d = Counter(sentences)
    l = list(d.values())
    max_repetitions_allowed = median(l) + stdev(l)
    sentences = [sentence for sentence in sentences if d[sentence] < max_repetitions_allowed]

    return sentences
