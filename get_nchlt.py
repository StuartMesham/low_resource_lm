import re
import argparse
import requests
import zipfile
import os
from io import BytesIO

# take --output_dir command-line argument
parser = argparse.ArgumentParser(description='Download NCHLT datasets.')
parser.add_argument('--output_dir', required=True, help='directory where output files will be saved')
args = parser.parse_args()

datasets = [
    (
        'https://repo.sadilar.org/bitstream/handle/20.500.12185/330/corpora.nchlt.nso.zip',  # url
        'nso/2.Corpora/CORP.NCHLT.nso.CLEAN.2.0.txt',  # file_name
        'sepedi.txt',  # output_name
    ),
    (
        'http://www.rma.nwu.ac.za/bitstream/handle/20.500.12185/321/corpora.nchlt.zu.zip',  # url
        'zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt',  # file_name
        'isizulu.txt',  # output_name
    )
]

for url, file_name, output_name in datasets:
    r = requests.get(url)
    zip = zipfile.ZipFile(BytesIO(r.content))
    corpus = zip.open(file_name).read().decode('utf-8').strip()
    
    # remove the first 11 lines (header)
    corpus = corpus.split('\n', 11)[11]
    
    # remove tags containing article filenames
    corpus = re.sub(r'<fn>.*</fn>', '', corpus)
    
    # put each sentence on a new line
    corpus = corpus.replace('. ', '.\n')
    
    # remove empty lines from corpus
    corpus = os.linesep.join([s for s in corpus.splitlines() if s.strip()])
    
    # write article to file (with each sentence on a new line)
    output_file_name = os.path.join(args.output_dir, output_name)
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write(corpus)
    
    print('total sentences in {}:'.format(output_name), corpus.count('\n'))
