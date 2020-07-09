import re
import argparse
import requests
import zipfile
import os
from io import BytesIO

import utils

# take --output_dir command-line argument
parser = argparse.ArgumentParser(description='Download Autshomato dataset (isiZulu and Sepedi).')
parser.add_argument('--output_dir', required=True, help='directory where output files will be saved')
args = parser.parse_args()

repo_urls = [
    'https://sourceforge.net/projects/autshumato/files/Corpora/ENG-ZUL.Release.zip/download',
]

sentence_count = 0

for url in repo_urls:
    print('processing:', url)
    r = requests.get(url)
    zip = zipfile.ZipFile(BytesIO(r.content))

    for name in zip.namelist():
        if '(eng-zul).zul' in name:
            article = zip.open(name).read().decode('utf-8').strip()
            sentence_count = article.count('\n')
            output_file_name = os.path.join(args.output_dir, os.path.basename(name))
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write(article)

print('total sentences:', sentence_count)
print("""
Autshomato datasets provided under Creative Commons Attribution Non-Commercial ShareAlike,
CTexT (Centre for Text Technology, North-West University), South Africa; Department of Arts and Culture, South Africa.
http://autshumato.sourceforge.net/ and http://www.nwu.ac.za/ctext
""")
