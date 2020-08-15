#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/scripts

echo 'fetching nchlt data'
mkdir -p data/nchlt
mkdir -p data/nchlt/isizulu
mkdir -p data/nchlt/sepedi
python3 scripts/get_nchlt.py --output_dir=data/nchlt

echo

echo 'fetching autshumato data'
mkdir -p data/autshumato
mkdir -p data/autshumato/isizulu
mkdir -p data/autshumato/sepedi
python3 scripts/get_autshumato.py --output_dir=data/autshumato

echo

echo 'fetching isolezwe data'
mkdir -p data/isolezwe
python3 scripts/get_isolezwe.py --output_dir=data/isolezwe

echo

echo 'Partitioning datasets'
python3 scripts/partition_datasets.py