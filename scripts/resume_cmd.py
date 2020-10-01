import os
from datetime import datetime

import torch
from transformers import Trainer, TrainingArguments

from gpt2_utils import evaluate_bpcs, MultilingualDataCollatorForLanguageModeling2, \
    MultilingualCachedTextDataset, TB_DIR, get_tokenizer
from layer_switching_gpt2 import GPT2LayerSwitchingLMHeadModel, LayerSwitchingGPT2Config

import argparse

LOG_DIR = 'logs'
MODEL_DIR = 'model_saves'

parser = argparse.ArgumentParser(description='Resume Training Run.')
parser.add_argument('--experiment_id', required=True)
parser.add_argument('--checkpoint_file', required=True)
parser.add_argument('--hparams', required=True)
args = parser.parse_args()

experiment_id = args.experiment_id
CHECKPOINT_FILE = args.checkpoint_file

hparams = eval(args.hparams)

tparams = {
    'max_steps': 200000,
    'patience': 4,
    'log_steps': 1000,
    'eval_steps': 5000,
    'save_steps': 5000,
    'num_workers': 0,
}


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
    assert 'num_workers' in tparams

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

    model.load_state_dict(torch.load(os.path.join(os.path.join(MODEL_DIR, experiment_id), CHECKPOINT_FILE + '/pytorch_model.bin')))

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

trainer = get_gpt2_trainer(experiment_id, hparams, tparams, disable_tqdm=False, disable_prediction_tqdm=False, log_to_console=True)

trainer.train(os.path.join(os.path.join(MODEL_DIR, experiment_id), CHECKPOINT_FILE))

val_metrics = evaluate_bpcs(
    trainer.tokenizers,
    trainer.model,
    hparams['val_data'],
    input_block_size=hparams['train_block_size'],
    stride=64,
    disable_tqdm=False,
)
# logger.info(repr(val_metrics))
print(repr(val_metrics))

test_metrics = evaluate_bpcs(
    trainer.tokenizers,
    trainer.model,
    hparams['test_data'],
    input_block_size=hparams['train_block_size'],
    stride=64,
    disable_tqdm=False,
)
# logger.info(repr(test_metrics))
print(repr(test_metrics))

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
