import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import accumulate
from math import log
from typing import List, Tuple
from uuid import uuid4

import torch
from filelock import FileLock
from tokenizers import ByteLevelBPETokenizer
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast, TrainingArguments, Trainer, is_torch_tpu_available, AdamW

from layer_switching_gpt2 import LayerSwitchingGPT2Config, GPT2LayerSwitchingLMHeadModel

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

CACHE_DIR = 'caches'
TB_DIR = 'logs/runs'
LOG_DIR = 'logs'
MODEL_DIR = 'model_saves'

logger = logging.getLogger(__name__)


class CachedTextDataset(Dataset):
    def __init__(
            self,
            tokenizer: GPT2TokenizerFast,
            dataset_files: [str],
            block_size: int,
            stride: int,
            shuffle: bool = False,
    ):

        self.examples = []

        for dataset_file in dataset_files:
            assert os.path.isfile(dataset_file)

            block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

            m = hashlib.md5()
            m.update(tokenizer.cache_id.encode())
            m.update(str(block_size).encode())
            m.update(str(stride).encode())
            m.update(dataset_file.encode())
            cache_id = m.hexdigest()

            cached_features_file = os.path.join(
                CACHE_DIR, 'dataset_{}'.format(cache_id),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file):
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        self.examples += pickle.load(handle)
                    logger.info(
                        f"Loaded features from {cached_features_file} [took %.3f s]", time.time() - start
                    )

                else:
                    start = time.time()

                    with open(dataset_file, encoding="utf-8") as f:
                        text = f.read()

                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    dataset_examples = []

                    for i in range(0, len(tokenized_text) - block_size + 1, stride):  # Truncate in block of block_size
                        tokens = tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                        dataset_examples.append(tokens)
                    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should look for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(dataset_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Created features for {dataset_file} [took %.3f s]", time.time() - start
                    )

                    self.examples += dataset_examples

        if shuffle:
            self.examples = [self.examples[i] for i in torch.randperm(len(self.examples)).tolist()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class MultilingualCachedTextDataset(Dataset):
    def __init__(
            self,
            datasets_info: List[Tuple[int, GPT2TokenizerFast, List[str]]],
            block_size: int,
            stride: int,
            batch_size: int,
            shuffle: bool = False,
    ):
        self.datasets = [
            CachedTextDataset(tokenizer, dataset_files, block_size, stride, shuffle=shuffle)
            for _, tokenizer, dataset_files in datasets_info
        ]

        self.batch_size = batch_size

        self.language_ids = [language_id for language_id, _, _ in datasets_info]

        self._lengths = [(len(dataset)+batch_size-1)//batch_size for dataset in self.datasets]
        self._cumulative_lengths = list(accumulate(self._lengths))

        self.length = sum(self._lengths)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        dataset_index = 0
        while self._cumulative_lengths[dataset_index] <= i:
            dataset_index += 1

        if dataset_index == 0:
            first_example_index = i * self.batch_size
        else:
            first_example_index = (i - self._cumulative_lengths[dataset_index-1]) * self.batch_size

        batch = self.datasets[dataset_index][first_example_index:first_example_index + self.batch_size]

        return {
            'language': self.language_ids[dataset_index],
            'input_ids': batch
        }


@dataclass
class MultilingualDataCollatorForLanguageModeling2:
    pad_token_id: int

    def __call__(self, batch: dict) -> dict:
        labels = batch['input_ids'].clone().detach()
        labels[labels == self.pad_token_id] = -100
        batch['labels'] = labels
        return batch



def get_tokenizer(train_data, vocab_size):
    assert vocab_size >= 257, 'vocab size must cover all possible bytes and one special token'

    m = hashlib.md5()
    m.update(str(vocab_size).encode())
    for file in train_data:
        m.update(file.encode())
    cache_id = m.hexdigest()

    cached_tokenizer_file = os.path.join(CACHE_DIR, 'tokenizer_{}'.format(cache_id))

    train_new_tokenizer = not os.path.exists(cached_tokenizer_file)
    if train_new_tokenizer:
        start = time.time()
        os.makedirs(cached_tokenizer_file)
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            train_data,
            vocab_size=vocab_size,
            special_tokens=['<|endoftext|>'],
            show_progress=False,
        )
        tokenizer.save_model(cached_tokenizer_file)
        logger.info(
            f"Trained tokenizer {cached_tokenizer_file} [took %.3f s]", time.time() - start
        )

    start = time.time()
    tokenizer = GPT2TokenizerFast.from_pretrained(cached_tokenizer_file)
    tokenizer.cache_id = cache_id

    if not train_new_tokenizer:
        logger.info(
            f"Loaded tokenizer from {cached_tokenizer_file} [took %.3f s]", time.time() - start
        )

    return tokenizer


def evaluate_bpcs(tokenizers, model, eval_data, input_block_size, stride, disable_tqdm=False):
    assert stride <= input_block_size
    metrics = {}
    for language_id, file_paths in eval_data:
        lls = []
        total_characters = 0
        if len(file_paths) > 1:
            logger.warning(f'You supplied multiple eval files for language {language_id}. Only the first one will be used.')
        with open(file_paths[0], 'r') as f:
            test_set = f.read()
        total_characters += len(test_set)
        encodings = tokenizers[language_id](test_set, return_tensors='pt')

        for i in tqdm(range(1, encodings.input_ids.size(1), stride), desc='Evaluating BPC', disable=disable_tqdm):
            begin_loc = max(i + stride - input_block_size, 0)
            end_loc = i + stride
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-stride] = -100

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    labels=target_ids,
                )
                # stride = number of tokens in the batch
                # outputs[0] = nats/token (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
                # outputs[0] * stride = nats
                log_likelihood = outputs[0].item() * stride

            lls.append(log_likelihood)

        # total nats / log(2) = total bits
        # total bits / total characters = bits/character
        metrics['bpc/' + file_paths[0]] = (sum(lls) / log(2)) / total_characters

    return metrics


def sanitise_hparams_for_tb(hparams):
    temp = hparams.copy()
    for k, v in temp.items():
        if type(v) == list:
            temp[k] = json.dumps(v)

    return temp


def default_hparams_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("hparam_tests", current_time + "_" + socket.gethostname())


def get_gpt2_trainer(experiment_id, hparams: dict, tparams: dict, disable_tqdm=False, disable_prediction_tqdm=True, log_to_console=False):
    assert 'vocab_size' in hparams
    assert 'train_data' in hparams
    assert 'val_data' in hparams
    assert 'test_data' in hparams
    assert 'model_max_input_size' in hparams
    assert 'pdrop' in hparams
    assert 'd_model' in hparams
    assert 'n_layers' in hparams
    assert 'n_heads' in hparams
    assert 'train_block_size' in hparams
    assert 'train_stride' in hparams
    assert 'val_block_size' in hparams
    assert 'val_stride' in hparams
    assert 'learning_rate' in hparams
    assert 'batch_size' in hparams
    assert 'optimizer' in hparams
    assert 'n_language_specific_attention_layers' in hparams
    assert 'n_languages' in hparams
    assert 'language_specific_input_embeds' in hparams
    assert 'language_specific_prediction_heads' in hparams
    assert 'semantic_concepts' in hparams
    assert 'language_specific_transformation' in hparams
    assert 'inverse_language_specific_transformation' in hparams
    assert 'tie_word_embeddings' in hparams
    assert 'tie_language_specific_transformation_weights' in hparams

    assert hparams['optimizer'] == 'adam', 'Only the adam optimizer is currently supported'
    if hparams['optimizer'] == 'SGD':
        assert 'momentum' in hparams

    assert 'max_steps' in tparams
    assert 'patience' in tparams
    assert 'log_steps' in tparams
    assert 'eval_steps' in tparams
    assert 'save_steps' in tparams

    config = LayerSwitchingGPT2Config(
        vocab_size=hparams['vocab_size'],
        n_positions=hparams['model_max_input_size'],
        n_ctx=hparams['model_max_input_size'],
        n_embd=hparams['d_model'],
        n_layer=hparams['n_layers'],
        n_head=hparams['n_heads'],
        n_language_specific_attention_layers=hparams['n_language_specific_attention_layers'],
        n_languages=hparams['n_languages'],
        language_specific_input_embeds=hparams['language_specific_input_embeds'],
        language_specific_prediction_heads=hparams['language_specific_prediction_heads'],
        semantic_concepts=hparams['semantic_concepts'],
        language_specific_transformation=hparams['language_specific_transformation'],
        inverse_language_specific_transformation=hparams['inverse_language_specific_transformation'],
        tie_language_specific_transformation_weights=hparams['tie_language_specific_transformation_weights'],
        attn_pdrop=hparams['pdrop'],
        embd_pdrop=hparams['pdrop'],
        resid_pdrop=hparams['pdrop'],
        summary_first_dropout=hparams['pdrop'],
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        tie_word_embeddings=hparams['tie_word_embeddings'],
    )

    model = GPT2LayerSwitchingLMHeadModel(config=config)

    hparams['total_trainable_parameters'] = model.num_parameters(only_trainable=True)

    if 'tokenizer_language_datasets' in hparams and 'tokenizer_dataset' in hparams:
        raise ValueError('You cannot specify both tokenizer_language_datasets and tokenizer_dataset.')
    elif 'tokenizer_language_datasets' in hparams:
        assert hparams['language_specific_input_embeds'] and hparams['language_specific_prediction_heads']
        assert len(hparams['tokenizer_language_datasets']) == len(hparams['train_data'])
        tokenizers = {
            language_id: get_tokenizer(train_files, hparams['vocab_size'])
            for language_id, train_files in hparams['tokenizer_language_datasets']
        }
    elif 'tokenizer_dataset' in hparams:
        # all languages use the same tokenizer
        tokenizer = get_tokenizer(hparams['tokenizer_dataset'], hparams['vocab_size'])
        tokenizers = {
            language_id: tokenizer
            for language_id, _ in hparams['train_data']
        }
    else:
        raise ValueError('You must specify either tokenizer_language_datasets or tokenizer_dataset.')

    train_dataset = MultilingualCachedTextDataset(
        [
            (language_id, tokenizers[language_id], dataset_files)
            for language_id, dataset_files in hparams['train_data']
        ],
        block_size=hparams['train_block_size'],
        stride=hparams['train_stride'],
        batch_size=hparams['batch_size'],
        shuffle=True
    )

    validation_dataset = MultilingualCachedTextDataset(
        [
            (language_id, tokenizers[language_id], dataset_files)
            for language_id, dataset_files in hparams['val_data']
        ],
        block_size=hparams['val_block_size'],
        stride=hparams['val_stride'],
        batch_size=hparams['batch_size'],
        shuffle=False,
    )

    data_collator = MultilingualDataCollatorForLanguageModeling2(
        list(tokenizers.values())[0].pad_token_id
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, experiment_id),
        logging_dir=os.path.join(TB_DIR, experiment_id),
        save_steps=tparams['save_steps'],
        max_steps=tparams['max_steps'],
        per_device_train_batch_size=1,
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
        logging_steps=tparams['log_steps'],
        eval_steps=tparams['eval_steps'],
        patience=tparams['patience'],
        prediction_loss_only=True,
        evaluate_during_training=True,
        disable_train_tqdm=disable_tqdm,
        disable_prediction_tqdm=disable_prediction_tqdm,
        hparams=hparams,
        is_dataset_pre_batched=True,
    )

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": hparams['weight_decay'],
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    #
    # if hparams['optimizer'] == 'adam':
    #     optimizer = AdamW(
    #         optimizer_grouped_parameters,
    #         lr=hparams['learning_rate'],
    #         betas=(0.9, 0.999),
    #         eps=1e-8,
    #     )
    # elif hparams['optimizer'] == 'SGD':
    #     assert 'momentum' in hparams
    #     optimizer = SGD(optimizer_grouped_parameters, lr=hparams['learning_rate'], momentum=hparams['momentum'])
    # else:
    #     raise ValueError(f'optimizer not recognised: {repr(hparams["optimizer"])}')
    #
    # if hparams['scheduler'] == 'reduce_on_plateau':
    #     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=hparams['scheduler_patience'])
    # elif hparams['scheduler'] == 'linear_with_warmup':
    #     scheduler = None  # configured automatically by Trainer class

    trainer = Trainer(
        tokenizers=tokenizers,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        log_to_console=log_to_console,
        # optimizers=(optimizer, scheduler),
    )

    return trainer


def run_experiment(hparams: dict, tparams: dict, eval_stride=64, disable_tqdm=True, disable_prediction_tqdm=True, log_to_console=False):
    experiment_id = uuid4().hex  # create a random experiment ID
    trainer = get_gpt2_trainer(experiment_id, hparams, tparams, disable_tqdm, disable_prediction_tqdm, log_to_console)
    trainer.train()
    val_metrics = evaluate_bpcs(
        trainer.tokenizers,
        trainer.model,
        hparams['val_data'],
        input_block_size=hparams['train_block_size'],
        stride=eval_stride,
        disable_tqdm=disable_prediction_tqdm,
    )
    logger.info(repr(val_metrics))

    test_metrics = evaluate_bpcs(
        trainer.tokenizers,
        trainer.model,
        hparams['test_data'],
        input_block_size=hparams['train_block_size'],
        stride=eval_stride,
        disable_tqdm=disable_prediction_tqdm,
    )
    logger.info(repr(test_metrics))

    log_data = {
        'id': experiment_id,
        'completion_time': datetime.now().strftime("%x %X"),
        'steps': trainer.global_step,
        'hparams': hparams,
        'tparams': tparams,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }

    with open(os.path.join(LOG_DIR, 'hparam_test.txt'), 'a') as logfile:
        logfile.write(repr(log_data) + '\n')
