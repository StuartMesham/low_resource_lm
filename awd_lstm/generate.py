###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

from utils import debug_print

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--seed_words', help='words to start with', default=None)
parser.add_argument('--debug', action='store_true', help='turn on debugging mode')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    if torch.cuda.is_available():
        model = torch.load(f)
    else:
        model = torch.load(f, map_location='cpu')
    print("Loaded model: " + args.checkpoint)
    model = model[0]
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(),
                 volatile=True)  # Random float 0-1 turned into a word id in the corpus dict
debug_print("Random initial word: " + corpus.dictionary.idx2word[input.item()], args)

if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    if args.seed_words:
        args.seed_words = args.seed_words.split()
        if len(args.seed_words) > 1:
            print("Not yet implemented for multiple seed words")
            args.seed_words = [args.seed_words[0]]
        #     TODO: Implement multiple seed word functionality

        debug_print("Seeding model with initial word:", args)
        debug_print(' '.join(args.seed_words), args)
        outf.write(' '.join(args.seed_words))
        input = Variable(torch.ones(1, 1).mul(corpus.dictionary.word2idx.get(args.seed_words[0])).long(),
                         volatile=True)  # TODO: use torch.no_grad() somehow
        print(args.seed_words[0])
        # for word in args.seed_words[1:]:
        #     output, hidden = model(input, hidden)
        #     word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

    word_idxs = []
    for i in range(args.words):
        output, hidden = model(input, hidden) # INSIDE: self.encoder is a encoder from the word_idx -> representation vector
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

        # Tensor where each row contains 1 index sampled from the multinomial probability distribution of word_weights
        # Samples as expected (higher weight is more likely)
        word_idx = torch.multinomial(word_weights, 1)[0] # CHECK: Can this produce anything except 0-400?
        word_idxs.append(word_idx) #Luc Code

        if args.debug:
            word_weight_argmax = word_weights.argmax().item()
            print("Word_weight argmax = " + str(word_weight_argmax) + " | " + corpus.dictionary.idx2word[word_weight_argmax]) # + " | " + str(word_weights.max().item()))
            print("Word_idx = " + str(word_idx) + " | " + corpus.dictionary.idx2word[word_idx]) # + " | " + )

        input.data.fill_(word_idx)  # set input.data = word_idx chosen
        word = corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))
        print(word, end=('\n' if i % 20 == 19 else ' '))  # TODO: Change this to actually have newlines
        if i % args.log_interval == 0:
            print('\n| Generated {}/{} words |'.format(i, args.words))
