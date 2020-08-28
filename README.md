# Low-Resource Language Modelling

Utilities for downloading and preprocessing training data.

# TODO:
- Fix <unk> and <eos>
- Add BPE
- Fix punctuation at end of line

# Notes from Jan:
- Splitcross = efficient softmax for large vocabularies. We should just use standard softmax with BPE
    - Not sure if you mean LogSoftmax or CrossEntropyLoss. Either way their splitcross only uses the "splits" on vocabularies larger than 75k
- Batch size is kept constant (10/1) for valid/test sets as you say.
- Want to setup tensorboard to help with visualising the changes in trining
- Need to look more at the model structure. I was trying to generate some text (to see if the model was training correctly) 
which it turns out had a lot of errors (it could only generate the first 400 words in the vocab). 
But taking another look at the model structure it seems like the decoder is **taking in more features than are output** by the last weightdrop layer?

- When calculating BPC, currently (I think) the model reports BPC as in cross entropy/token average. Can we just take the total CE of the test set/nCharacters instead and call it approcimate bpc?

- Comparisons should be standalone, add QRNN, Morgrifier maybe, and baseline LSTM from pytorch


## Usage example:
Ensure that all scripts are run from the root directory, <b>not /scripts/</b>

install requirements:
```bash
pip3 install -r requirements.txt
```

fetch training data:
```bash
./scripts/fetch_data.sh
```

## AWD-LSTM Acknowledgements
Code accessed from https://github.com/salesforce/awd-lstm-lm 
Cited as requested:
```
@article{merityRegOpt,
  title={{Regularizing and Optimizing LSTM Language Models}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1708.02182},
  year={2017}
}
```

```
@article{merityAnalysis,
  title={{An Analysis of Neural Language Modeling at Multiple Scales}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1803.08240},
  year={2018}
}
```
See the readme at /awd_lstm/README.md for further details