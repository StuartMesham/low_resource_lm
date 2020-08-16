import os
import argparse
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

parser = argparse.ArgumentParser(description='Train and Save a GPT-2 Model.')
parser.add_argument('--output_dir', required=True, help='directory where model files will be saved')
parser.add_argument('--train_data', required=True, help='text file containing training data')
parser.add_argument('--validation_data', required=True, help='text file containing validation data')
parser.add_argument('--train_block_size', default=32, type=int, required=False, help='size of each training example')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='training batch size')
parser.add_argument('--steps', default=4000, type=int, required=False, help='number of training steps to perform')
parser.add_argument('--vocab_size', default=5000, type=int, required=False, help='BPE vocabulary size')
parser.add_argument('--d_model', default=128, type=int, required=False, help='dimensionality of the hidden layers of the model')
parser.add_argument('--n_layers', default=4, type=int, required=False, help='number of hidden layers in the model')
parser.add_argument('--n_heads', default=4, type=int, required=False, help='number of heads in the model')
parser.add_argument('--model_max_input_size', default=1024, type=int, required=False, help='dimensionality of the hidden layers of the model')
parser.add_argument('--learning_rate', default=1e-3, type=float, required=False, help='initial learning rate for adam optimizer')
parser.add_argument('--pdrop', default=0.1, type=float, required=False, help='dropout probability')

args = parser.parse_args()

# train tokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    args.train_data,
    vocab_size=args.vocab_size,
    special_tokens=['<|endoftext|>'],
    show_progress=False,
)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

tokenizer.save_model(args.output_dir)

# train model

tokenizer = GPT2TokenizerFast.from_pretrained(args.output_dir)

config = GPT2Config(
    vocab_size=args.vocab_size,
    n_positions=args.model_max_input_size,
    n_ctx=args.model_max_input_size,
    n_embd=args.d_model,
    n_layer=args.n_layers,
    n_head=args.n_heads,
    attn_pdrop=args.pdrop,
    embd_pdrop=args.pdrop,
    resid_pdrop=args.pdrop,
    summary_first_dropout=args.pdrop,
    bos_token_id=0,
    eos_token_id=0,
    pad_token_id=0,
)

model = GPT2LMHeadModel(config=config)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=args.train_data,
    block_size=args.train_block_size,
)

validation_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=args.validation_data,
    block_size=args.train_block_size,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    max_steps=args.steps,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    save_steps=1000,
    logging_steps=100,
    eval_steps=100,
    save_total_limit=3,
    evaluate_during_training=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    prediction_loss_only=True,
)

trainer.save_model(args.output_dir)  # save so we can see the config file for aborted runs

trainer.train()

trainer.save_model(args.output_dir)
