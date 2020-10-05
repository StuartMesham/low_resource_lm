# Low-Resource Language Modelling

An implementation of a multilingual GPT-2 and utilities for downloading and preprocessing training data.

## Usage example
Ensure that all scripts are run from the root directory.

install requirements:
```bash
pip3 install -r requirements.txt
```

add scripts directory to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`/scripts
```

fetch training data:
```bash
./scripts/fetch_data.sh
```

train multilingual isiZulu GPT-2 model on all languages:
```bash
python3 scripts/train_example.py
```

view tensorboard logs:
```bash
tensorboard --logdir=logs/runs
```

generate results CSV from `logs/experiment_logs.txt`:
```bash
python3 scripts/create_csv.py
```

The results of the experiment can now be viewed in `logs/results.csv`.

## Notes

The `scripts/fetch_data.sh` script downloads, preprocesses and partitions all required data into test, train and validation splits.

The `scripts/gpt2_utils.py` module contains the `run_experiment` method which trains, evaluates and logs model results for an input set of hyper-parameters. The run_experiment method can also be used to resume the training by supplying the checkpoint-dir argument to load.

The run experiment logs the results to a file. By default, the file is in `logs/experiment_logs.txt`. Each line in the file is a string representation of a python dictionary containing hyper-parameters, training parameters (e.g. max training steps) and evaluation results of one run. The `scripts/create_csv.py` script reads this log file and outputs a CSV with the same information for ease of analysis. During experimentation, this increased the flexibility of the logging system over simply logging directly to a CSV since new parameters and metrics could added and removed without having to modify previous logs. By default, the CSV file is saved to `logs/results.csv`.

## HuggingFace Library

This implementation relies on the [HuggingFace transformers](https://github.com/huggingface/transformers) library.
We use a custom fork with the following modifications:

1. An early stopping feature added to the Trainer class as per [this pull request](https://github.com/huggingface/transformers/pull/4186).
2. Bits-per-character evaluation during training added to Trainer class.
3. Minor modifications to Trainer and TrainingArguments classes for compatibility with custom data loading code used to enable multilingual training with language specific weights.

These modifications can be inspected in `transformers/src/transformers/trainer` and `transformers/src/transformers/training_args` in the HuggingFace fork.