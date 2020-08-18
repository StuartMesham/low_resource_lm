from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import math
import argparse

parser = argparse.ArgumentParser(description='Evaluate a trained GPT-2 model on a test set.')
parser.add_argument('--model_dir', required=True, help='directory where model files are saved')
parser.add_argument('--test_set', required=True, help='file containing test set')
parser.add_argument('--input_block_size', default=32, type=int, required=False, help='size of each input example')
args = parser.parse_args()


tokenizer = GPT2TokenizerFast.from_pretrained(args.model_dir)
model = GPT2LMHeadModel.from_pretrained(args.model_dir)

validation_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=args.test_set,
    block_size=args.input_block_size,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=args.model_dir,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=validation_dataset,
    prediction_loss_only=True,
)

eval_loss = trainer.evaluate()['eval_loss']
print('eval_loss:', eval_loss)
print('perplexity:', math.exp(eval_loss))
