import hashlib
import json
import os
import logging
import time
import pickle
from typing import List, Tuple
from filelock import FileLock
import torch
from torch.utils.data.dataset import Dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from multilingual_data_collator import MultilingualDataCollatorForLanguageModeling

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CACHE_DIR = 'caches'

logger = logging.getLogger(__name__)


class MultilingualTextDataset(Dataset):
    def __init__(
            self,
            tokenizer: GPT2TokenizerFast,
            dataset_files: List[Tuple[int, str]],
            block_size: int,
            use_token_type_ids=True,
    ):

        self.examples = []
        for language_id, file_path in dataset_files:
            assert os.path.isfile(file_path)

            block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

            m = hashlib.md5()
            m.update(tokenizer.cache_id.encode())
            m.update(str(block_size).encode())
            m.update(file_path.encode())
            cache_id = m.hexdigest()

            cached_features_file = os.path.join(
                CACHE_DIR, 'dataset_{}'.format(cache_id),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):
                language_examples = []

                if os.path.exists(cached_features_file):
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        language_examples = pickle.load(handle)
                    logger.info(
                        f"Loaded features from {cached_features_file} [took %.3f s]", time.time() - start
                    )

                else:
                    start = time.time()

                    with open(file_path, encoding="utf-8") as f:
                        text = f.read()

                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                        tokens = tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                        language_examples.append(tokens)
                    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should look for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(language_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Created features for {file_path} [took %.3f s]", time.time() - start
                    )
                if use_token_type_ids:
                    # add language IDs
                    self.examples += [[tokens, [language_id]*len(tokens)] for tokens in language_examples]
                else:
                    self.examples += tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def get_tokenizer(train_data, vocab_size):
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


def evaluate_bpc(tokenizer, model, eval_data, input_block_size, stride):
    lls = []
    total_characters = 0
    for language_id, test_set in eval_data:
        with open(test_set, 'r') as f:
            test_set = f.read()
        total_characters += len(test_set)
        encodings = tokenizer(test_set, return_tensors='pt')

        for i in tqdm(range(1, encodings.input_ids.size(1), stride)):
            begin_loc = max(i + stride - input_block_size, 0)
            end_loc = i + stride
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-stride] = -100

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    token_type_ids=torch.full(input_ids.size(), language_id, dtype=torch.int64, device=device) if model.config.type_vocab_size is not None else None,
                    labels=target_ids,
                )
                log_likelihood = outputs[0] * stride

            lls.append(log_likelihood)

    return torch.pow(2, torch.stack(lls).sum() / total_characters).item()


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


def get_gpt2_trainer(hparams: dict, tparams: dict, disable_tqdm=True):
    assert 'tokenizer_train_data' in hparams
    assert 'vocab_size' in hparams
    assert 'train_data' in hparams
    assert 'val_data' in hparams
    assert 'model_max_input_size' in hparams
    assert 'pdrop' in hparams
    assert 'd_model' in hparams
    assert 'n_layers' in hparams
    assert 'n_heads' in hparams
    assert 'train_block_size' in hparams
    assert 'learning_rate' in hparams
    assert 'batch_size' in hparams
    assert 'use_token_type_ids' in hparams

    assert 'max_steps' in tparams
    assert 'patience' in tparams
    assert 'eval_steps' in tparams

    tokenizer = get_tokenizer(hparams['tokenizer_train_data'], hparams['vocab_size'])

    config = GPT2Config(
        vocab_size=hparams['vocab_size'],
        type_vocab_size=len(hparams['train_data']) if hparams['use_token_type_ids'] else None,
        n_positions=hparams['model_max_input_size'],
        n_ctx=hparams['model_max_input_size'],
        n_embd=hparams['d_model'],
        n_layer=hparams['n_layers'],
        n_head=hparams['n_heads'],
        attn_pdrop=hparams['pdrop'],
        embd_pdrop=hparams['pdrop'],
        resid_pdrop=hparams['pdrop'],
        summary_first_dropout=hparams['pdrop'],
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )

    model = GPT2LMHeadModel(config=config)

    train_dataset = MultilingualTextDataset(
        tokenizer=tokenizer,
        dataset_files=hparams['train_data'],
        block_size=hparams['train_block_size'],
        use_token_type_ids=hparams['use_token_type_ids'],
    )

    validation_dataset = MultilingualTextDataset(
        tokenizer=tokenizer,
        dataset_files=hparams['val_data'],
        block_size=hparams['train_block_size'],
        use_token_type_ids=hparams['use_token_type_ids'],
    )

    if hparams['use_token_type_ids']:
        data_collator = MultilingualDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    training_args = TrainingArguments(
        output_dir='',
        save_steps=0,
        max_steps=tparams['max_steps'],
        per_device_train_batch_size=hparams['batch_size'],
        learning_rate=hparams['learning_rate'],
        logging_steps=100,
        eval_steps=tparams['eval_steps'],
        patience=tparams['patience'],
        evaluate_during_training=True,
        disable_tqdm=disable_tqdm,
    )

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        prediction_loss_only=True,
    )

    return trainer
