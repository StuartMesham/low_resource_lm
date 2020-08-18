# Low-Resource Language Modelling

Utilities for downloading and preprocessing training data.

# TODO:
- Fix <unk> and <eos>
- Add BPE
- Fix punctuation at end of line

## Usage example:
Ensure that all scripts are run from the root directory, <b>not /scripts/</b>
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