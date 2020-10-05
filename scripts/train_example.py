# Written by Stuart Mesham (MSHSTU001)

import sys
import logging
from gpt2_utils import run_experiment

# print logs to console
streamHandler = logging.StreamHandler(sys.stdout)
streamHandler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger = logging.getLogger('gpt2_utils')
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

tokenizer_train_data_nguni = [
    'data/nchlt/isizulu_train.txt',
    'data/nchlt/isixhosa_train.txt',
    'data/nchlt/siswati_train.txt',
    'data/nchlt/isindebele_train.txt',
]

tokenizer_train_data_sotho_tswana = [
    'data/nchlt/sepedi_train.txt',
    'data/nchlt/setswana_train.txt',
    'data/nchlt/sesotho_train.txt',
]

tokenizer_train_data_tswa_ronga = [
    'data/nchlt/xitsonga_train.txt',
]

tokenizer_train_data_venda = [
    'data/nchlt/tshivenda_train.txt',
]

tokenizer_train_data_all = \
    tokenizer_train_data_nguni \
    + tokenizer_train_data_sotho_tswana \
    + tokenizer_train_data_tswa_ronga \
    + tokenizer_train_data_venda

val_data = [
    (0, ['data/nchlt/isizulu_valid.txt']),
]

test_data = [
    (0, ['data/nchlt/isizulu_test.txt']),
]

hparams = {
    'tokenizer_dataset': tokenizer_train_data_nguni,
    'vocab_size': 8000,
    'train_data': [  # first value of tuple indicates the index of the set of language/family specific weights if used.
        (0, tokenizer_train_data_nguni),
        (1, tokenizer_train_data_sotho_tswana),
        (2, tokenizer_train_data_tswa_ronga),
        (3, tokenizer_train_data_venda)
    ],
    'val_data': val_data,
    'test_data': test_data,
    'model_max_input_size': 1024,
    'pdrop': 0.1,
    'd_model': 32,
    'n_layers': 8,
    'n_heads': 8,
    'train_block_size': 128,
    'train_stride': 16,
    'val_block_size': 128,
    'val_stride': 128,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'weight_decay': 0.1,
    'scheduler': 'linear_with_warmup',
    'n_language_specific_attention_layers': 0,
    'n_languages': 4,  # number of sets of language/family specific layers
    'language_specific_input_embeds': False,
    'language_specific_prediction_heads': False,
    'language_specific_transformation': False,
    'inverse_language_specific_transformation': False,
    'semantic_concepts': None,
    'tie_word_embeddings': True,
}

tparams = {
    'max_steps': 10,
    'patience': 4,
    'log_steps': 1,
    'eval_steps': 5,
    'save_steps': 2,
}

run_experiment(hparams, tparams, eval_stride=64, experiment_id='small_isizulu_example')
