import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from datetime import datetime
import os

import data
import model

from utils import batchify, get_batch, repackage_hidden, debug_print, get_basic_batch

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU, BASIC)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--vocab_size', default=5000, help='size of vocab ONLY IF using bpe')
parser.add_argument('--use_bpe', default=True, help='use huggingface byte level bpe tokenizer')
parser.add_argument('--early_exit', default=False,
                    help='Exit early from model training once valid_loss is not changing enough per run')
parser.add_argument('--descriptive_name', default='', help='Descriptive tag to add to the tensorboard save details.')
parser.add_argument('--log_hparams_only', default=False,
                    help='Skip training and jump straight to logging validation score for hparams metrics')
args = parser.parse_args()
args.tied = True
run_name = str(args.data).replace('/', '-') + "/" + args.model + "/" + datetime.now().strftime(
    "%d|%H:%M") + "_" + args.descriptive_name
drive_name = "/content/drive/My Drive/Colab Notebooks/runs/"
writer = SummaryWriter(drive_name + run_name)
sargs = ''
for arg in vars(args):
    sargs += ("{:<16}: {}  \n".format(str(arg), str(getattr(args, arg))))
if not args.log_hparams_only: writer.add_text('args', sargs)
print(sargs)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        if torch.cuda.is_available():
            model, criterion, optimizer = torch.load(f)
        else:
            model, criterion, optimizer = torch.load(f, map_location='cpu')


import os
import hashlib

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data, args.vocab_size, args.use_bpe)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
print(corpus.dictionary.avg_characters_per_token)
# CHECK: Why is validation batching not the same as testing/training
###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss

criterion = None  # CHECK: Could change this for the standard pytorch cross entropy loss

ntokens = len(corpus.dictionary)

def basic_train():
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    params = list(model.parameters()) + list(criterion.parameters())

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # Fairly certain we can make (hidden, state_c) = hidden
        hidden, state_c = model.init_state(args.batch_size)
        batch, i = 0, 0

        while i < train_data.size(0) - 1 - 1:
            bptt = args.bptt
            optimizer.zero_grad()

            data, targets = get_basic_batch(train_data, i, args, seq_len=bptt)

            y_pred, (hidden, state_c) = model(data, (hidden, state_c))
            loss = criterion(y_pred.transpose(1,2), targets)

            hidden = hidden.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})

            batch += 1
            i += bptt

if args.model == 'BASIC':
    model = model.BasicModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
    basic_train()

else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                           args.dropouti, args.dropoute, args.wdrop, args.tied)
# writer.add_graph(model, )
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, model.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop

        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = args.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)
print(model)
if not args.log_hparams_only: writer.add_text('model_structure',
                                              "Total Params: " + str(total_params) + "  \n" + str(model).replace('\n',
                                                                                                                 '  \n'))


###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):  # Jump forwards in bptt (70) increments
        data, targets = get_batch(data_source, i, args,
                                  evaluation=True)  # Gets the data and the target data to be produced
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)






def train():
    global writer
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(
            args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss),
                              cur_loss / math.log(2)))  # TODO: WRONG! Need to divide again by characters/token
            writer.add_scalar('train/loss', cur_loss, (epoch - 1) * (len(train_data) // args.bptt) + batch)
            writer.add_scalar('train/ppl', math.exp(cur_loss), (epoch - 1) * (len(train_data) // args.bptt) + batch)
            writer.add_scalar('train/bpc (token)', cur_loss / math.log(2),
                              (epoch - 1) * (len(train_data) // args.bptt) + batch)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
old_loss = 100000000
new_loss = 100000000
# # Run on test data.
# test_loss = evaluate(test_data, test_batch_size)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
#     test_loss, math.exp(test_loss), test_loss / math.log(2)))  # NOTE: Ask Jan about bpc here
# print('=' * 89)  # NOTE: NOT BPC but rather token level cross entropy etc, can I just divide by avg token length
#
#


if not args.log_hparams_only:
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)
                writer.add_scalar('valid/loss', val_loss2, epoch)
                writer.add_scalar('valid/ppl', math.exp(val_loss2), epoch)
                writer.add_scalar('valid/bpc (token)', val_loss2 / math.log(2), epoch)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(val_data, eval_batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)
                writer.add_scalar('valid/loss', val_loss, epoch)
                writer.add_scalar('valid/ppl', math.exp(val_loss), epoch)
                writer.add_scalar('valid/bpc (token)', val_loss / math.log(2), epoch)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0.,
                                                 weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

            # if args.early_exit and :

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
print('Loaded best saved model')

if args.log_hparams_only:
    stored_loss = evaluate(val_data, eval_batch_size)
writer.add_hparams(args.__dict__,
                   {'hparam/val_loss': stored_loss,
                    'hparam/val_bpc': stored_loss / math.log(2) / corpus.dictionary.avg_characters_per_token.get(
                        'valid')})

print("Evaluating on test data...")
# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)

print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss),
    test_loss / math.log(2) / corpus.dictionary.avg_characters_per_token.get('test')))  # NOTE: Ask Jan about bpc here

writer.add_scalar('test/loss', test_loss, 0)
writer.add_scalar('test/ppl', math.exp(test_loss), 0)
writer.add_scalar('test/bpc', test_loss / math.log(2) / corpus.dictionary.avg_characters_per_token.get('test'), 0)
print('=' * 89)
