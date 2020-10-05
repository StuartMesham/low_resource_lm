# Low-Resource Language Modelling
Includes the necessary commandline arguments for running each of the models as well as 
utilities for downloading and preprocessing training data and instructions of how to install the dependencies.

Included is a copy of a iPython notebook from google Colab which will allow for easy running of the project code.


## Usage example:
Ensure that all scripts are run from the root directory, <b>not /scripts/</b>

Models require the same pytorch/CUDA versions as required by the original AWD-LSTM library (see awd_lstm/README.md)

Install requirements:
```bash
pip3 install -r requirements.txt
```
If using QRNN models, also install the following:
```bash
pip3 install cupy pynvrtc git+https://github.com/saurabh3949/pytorch-qrnn
```

Fetch training data:
```bash
./scripts/fetch_data.sh
```

The minimum arguments to run the program are:
```bash
python3 awd_lstm/main.py \
    --data data/nchlt/isizulu/ \
    --save "/content/drive/My Drive/Colab Notebooks/nchlt_zulu_bpe_ptbInspired.pt" \
    --descriptive_name "ptbInspired" \
```

## Experiments
The following provide the needed parameters to recreate the top performing QRNN, AWD-LSTM and basic LSTM models on the NCHLT-isiZulu dataset.
To run on alternate datasets the --data argument should be changed. Each of the models takes at least 3-4 hours to reach adequate performance and up to 10-12 to reach the best performance.
Models were trained using a mix of Nvidia K80, P100 and V100 GPUs.

### AWD-LSTM
```bash
python3 -u awd_lstm/main.py \
    --save "AWD_LSTM_Test.pt" \
    --descriptive_name "ExampleAWDLSTM" \
    --data data/nchlt/isizulu/ \
    --model "LSTM" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 80 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --nonmono 8 \

```

### QRNN
```bash
python -u awd_lstm/main.py \
    --dropouth 0.2 \
    --seed 1882 \
    --epoch 500 \
    --emsize 800 \
    --nonmono 8 \
    --clip 0.25 \
    --dropouti 0.4 \
    --dropouth 0.2 \
    --nhid 1550 \
    --nlayers 4 \
    --wdrop 0.1 \
    --batch_size 40 \
    --data data/nchlt/isizulu/ \
    --model QRNN \
    --save "QRNN_test.pt" \
    --descriptive_name "ExampleQRNN" 
```

### Basic LSTM
```bash
python3 -u awd_lstm/main.py \
    --save "basicInputDrop.pt" \
    --descriptive_name "basicInputDrop_example" \
    --data data/nchlt/isizulu/ \
    --dropouti 0.25 \
    --model LSTM \
    --emsize 400 \
    --nhid 1550 \
    --nlayers 1 \
    --lr 5.0 \
    --clip 0.0 \
    --epochs 500 \
    --batch_size 40 \
    --bptt 70 \
    --dropout 0.0 \
    --dropouth 0.0 \
    --dropoute 0.0 \
    --wdrop 0.0 \
    --seed 4002 \
    --nonmono 5 \
    --alpha 0.0 \
    --beta 0.0 \
    --wdecay 0.0 \
    --chpc True \
```


## AWD-LSTM Acknowledgements
Code accessed from https://github.com/salesforce/awd-lstm-lm 

See the readme at /awd_lstm/README.md for further details

Merity, Stephen et al. "Regularizing and Optimizing LSTM Language Models". arXiv preprint arXiv:1708.02182. (2017).

Merity, Stephen et al. "An Analysis of Neural Language Modeling at Multiple Scales". arXiv preprint arXiv:1803.08240. (2018).
