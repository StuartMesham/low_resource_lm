import re
import argparse
import requests
import zipfile
import os
from io import BytesIO
import utils

# take --output_dir command-line argument
parser = argparse.ArgumentParser(description='Download isolezwe dataset (isiZulu).')
parser.add_argument('--output_dir', required=True, default='data/isolezwe',
                    help='directory where output files will be saved')
args = parser.parse_args()

repo_urls = [
    'https://codeload.github.com/newstools/2016-iol-isolezwe/zip/master',
    'https://codeload.github.com/newstools/2017-iol-isolezwe/zip/master',
    'https://codeload.github.com/newstools/2018-iol-isolezwe/zip/master',
    'https://codeload.github.com/newstools/2019-iol-isolezwe/zip/master',
    'https://codeload.github.com/newstools/2020-iol-isolezwe/zip/master',
]

sentence_count = 0
corpus = []

for url in repo_urls:
    print('processing:', url)
    r = requests.get(url)
    zip = zipfile.ZipFile(BytesIO(r.content))

    for name in zip.namelist():
        if not name.endswith('/'):
            article = zip.open(name).read().decode('utf-8').strip()

            # fix cases where they don't have spaces between sentences
            # TODO Jan: How should we clean end of sentences
            article = re.sub(r'\.(?! )', '. ', article)
            article = re.sub(r'\!(?! )', '! ', article)
            article = re.sub(r'\?(?! )', '? ', article)

            # remove tag that is sometimes repeated
            article = re.sub(r'&.*;', '', article)

            # remove brackets with numbers in them (they seemed to appear often)
            article = re.sub(r'\(\d*\)', '', article)

            # TODO Jan: How are we tokenizing
            # remove extra whitespace
            article = re.sub('\\s+', ' ', article)

            # replace strange quote character
            article = utils.normalize_quote_characters(article)

            # discard articles with imbalanced quotes
            if article.count('"') % 2 != 0:
                continue

            # split article into array of sentences
            # regex help from https://stackoverflow.com/questions/11502598/how-to-match-something-with-regex-that-is-not-between-two-special-characters
            sentences = re.split('(?<=\.|\!|\?) (?=(?:[^"]*"[^"]*")*[^"]*\Z)', article)

            # un-capitalize first words in articles
            sentences = [re.sub(r'[A-Z -]+', lambda m: m.group().capitalize(), sentence, count=1) for sentence in sentences]

            # update total sentence count
            sentence_count += len(sentences)
            corpus = corpus + sentences

corpus = utils.clean_sentences(corpus, min_length=16, illegal_substrings=['@', '%2'])

print('total sentences:', sentence_count)

with open(os.path.join(args.output_dir, os.path.basename("isizulu.txt")), 'w', encoding='utf-8') as f:
    f.write('\n'.join(corpus)+'\n')
