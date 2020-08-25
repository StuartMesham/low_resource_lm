# https://huggingface.co/transformers/perplexity.html
import os
import argparse
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

parser = argparse.ArgumentParser(description='Evaluate a trained GPT-2 model on a test set.')
parser.add_argument('--model_dir', required=True, help='directory where model files are saved')
parser.add_argument('--test_set', required=True, help='file containing test set')
parser.add_argument('--input_block_size', default=32, type=int, required=False, help='size of each input example')
parser.add_argument('--stride', default=512, type=int, required=False, help='size of each input example')
args = parser.parse_args()

model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(args.model_dir)

f = open(args.test_set, 'r')
test_set = f.read()
f.close()

encodings = tokenizer(test_set, return_tensors='pt')

lls = []
for i in tqdm(range(1, encodings.input_ids.size(1), args.stride)):
    begin_loc = max(i + args.stride - args.input_block_size, 0)
    end_loc = i + args.stride
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-args.stride] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * args.stride

    lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / encodings.input_ids.size(1)).item()
bpc = torch.pow(2, torch.stack(lls).sum() / len(test_set)).item()
print('ppl:', ppl)
print('bpc:', bpc)
