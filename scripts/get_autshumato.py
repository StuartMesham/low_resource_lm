import argparse
import requests
import zipfile
import os
from io import BytesIO

# take --output_dir command-line argument
parser = argparse.ArgumentParser(description='Download Autshumato dataset (isiZulu and Sepedi).')
parser.add_argument('--output_dir', required=True, help='directory where output files will be saved')
args = parser.parse_args()

datasets = [
    (
        'https://master.dl.sourceforge.net/project/autshumato/Corpora/ENG-ZUL.Release.zip',  # url
        'lcontent.DACB.DataVirVrystellingOpWeb.(eng-zul).zul.1.0.0.CAM.2010-09-23.txt',  # file_name
        'isizulu.txt',  # output_name
    ),
    (
        'https://master.dl.sourceforge.net/project/autshumato/Corpora/ENG-NSO.Release.zip',  # url
        'lcontent.DACB.DataVirVrystellingOpWeb.(eng-nso).nso.1.0.0.CAM.2010-09-23.txt',  # file_name
        'sepedi.txt',  # output_name
    )
]

for url, file_name, output_name in datasets:
    print('processing:', url)
    r = requests.get(url)
    zip = zipfile.ZipFile(BytesIO(r.content))
    corpus = zip.open(file_name)
    corpus = corpus.read()
    corpus = corpus.decode('utf-8')
    corpus = corpus.strip()
    corpus = corpus.replace('\r\n', '\n')

    # remove the first 2607 lines (Transcription of constitution with poor formatting)
    if file_name is 'isizulu.txt':
        corpus = corpus.split('\n', 2607)[2607]
    else:
        corpus = corpus.split('\n', 5302)[5302] # Could also use something earlier

    output_file_name = os.path.join(args.output_dir, output_name)
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write(corpus)

    print('total sentences in {}:'.format(output_name), corpus.count('\n'))

# print('Autshumato datasets provided under Creative Commons Attribution Non-Commercial ShareAlike, '
#       'CTexT (Centre for Text Technology, North-West University), South Africa; '
#       'Department of Arts and Culture, South Africa. '
#       'http://autshumato.sourceforge.net/ and http://www.nwu.ac.za/ctext')
