#  Written in part by both Stuart Mesham (MSHSTU001) and Luc Hayward (HYWLUC001)

import re
import argparse
import requests
import zipfile
import os
from io import BytesIO
import utils

# take --output_dir command-line argument
parser = argparse.ArgumentParser(description='Download NCHLT datasets.')
parser.add_argument('--output_dir', required=True, help='directory where output files will be saved')
args = parser.parse_args()

datasets = [
    (
        'https://repo.sadilar.org/bitstream/handle/20.500.12185/330/corpora.nchlt.nso.zip',  # url
        'nso/2.Corpora/CORP.NCHLT.nso.CLEAN.2.0.txt',  # file_name
        'sepedi/sepedi.txt',  # output_name
        11,  # lines_to_remove
    ),
    (
        'http://www.rma.nwu.ac.za/bitstream/handle/20.500.12185/321/corpora.nchlt.zu.zip',  # url
        'zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt',  # file_name
        'isizulu/isizulu.txt',  # output_name
        11,  # lines_to_remove
    ),
    # (
    #     'https://repo.sadilar.org/bitstream/handle/20.500.12185/314/corpora.nchlt.xh.zip',  # url
    #     'xh/2.Corpora/CORP.NCHLT.xh.CLEAN.2.0.txt',  # file_name
    #     'isixhosa.txt',  # output_name
    #     11,  # lines_to_remove
    # ),
    # (
    #     'https://repo.sadilar.org/bitstream/handle/20.500.12185/364/corpora.nchlt.ts.zip',  # url
    #     'ts/2.Corpora/CORP.NCHLT.ts.CLEAN.2.0.txt',  # file_name
    #     'xitsonga.txt',  # output_name
    #     11,  # lines_to_remove
    # ),
    # (
    #     'http://www.rma.nwu.ac.za/bitstream/handle/20.500.12185/343/corpora.nchlt.tn.zip',  # url
    #     'tn/2.Corpora/CORP.NCHLT.tn.CLEAN.2.0.txt',  # file_name
    #     'setswana.txt',  # output_name
    #     11,  # lines_to_remove
    # ),
    # (
    #     'http://www.rma.nwu.ac.za/bitstream/handle/20.500.12185/348/corpora.nchlt.ss.zip',  # url
    #     'ss/2.Corpora/CORP.NCHLT.ss.CLEAN.2.0.txt',  # file_name
    #     'siswati.txt',  # output_name
    #     11,  # lines_to_remove
    # ),
    # (
    #     'https://repo.sadilar.org/bitstream/handle/20.500.12185/308/corpora.nchlt.nr.zip',  # url
    #     'nr/2.Corpora/CORP.NCHLT.nr.CLEAN.2.0.txt',  # file_name
    #     'isindebele.txt',  # output_name
    #     11,  # lines_to_remove
    # ),
    # (
    #     'http://www.rma.nwu.ac.za/bitstream/handle/20.500.12185/357/corpora.nchlt.ve.zip',  # url
    #     've/2.Corpora/CORP.NCHLT.ve.CLEAN.2.0.txt',  # file_name
    #     'tshivenda.txt',  # output_name
    #     312,  # lines_to_remove
    # ),
]

for url, file_name, output_name, lines_to_remove in datasets:
    print('processing:', url)

    r = requests.get(url)
    zip = zipfile.ZipFile(BytesIO(r.content))
    corpus = zip.open(file_name).read().decode('utf-8').strip()

    # remove tags containing article filenames
    corpus = re.sub(r'<fn>.*</fn>', '', corpus)
    
    # put each sentence on a new line
    corpus = corpus.replace('. ', '.\n')
    
    # remove empty lines from corpus
    sentences = corpus.splitlines()

    sentences = utils.clean_sentences(
        sentences,
        illegal_substrings=['\ufeff', '='],
        lines_to_remove=lines_to_remove,
    )

    # write article to file (with each sentence on a new line)
    output_file_name = os.path.join(args.output_dir, output_name)
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    
    print('total sentences in {}:'.format(output_name), corpus.count('\n'))
