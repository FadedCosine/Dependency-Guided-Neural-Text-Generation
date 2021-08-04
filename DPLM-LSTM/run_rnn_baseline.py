import argparse
import time
import math
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from functools import partial
import data
from ONLSTM_model import ONLSTMModel
from AWDLSTM_model import RNNModel
from tqdm import tqdm
from utils import batchify, batchify_dep_tokenlist, get_batch, repackage_hidden, collate_func_for_tok
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import os
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def model_save(args, fn, model, criterion, optimizer):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(args, fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer

###############################################################################
# Training code
###############################################################################

def evaluate(args, data_source, model, corpus, criterion, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.rnnmodel == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        # input = batch['net_input']['src_tokens'].to(args.device).t()
        # targets = batch['target'].to(args.device).t()
        input, targets = get_batch(data_source, i, args)
        # print("targets is : ", [ corpus.dictionary.idx2word[id] for id in targets])
        output, hidden = model(input, hidden)
        total_loss += len(input) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)


def train_a_epoch(args, train_dataset, model, corpus, optimizer, criterion, params, epoch):
    # Turn on training mode which enables dropout.
    if args.rnnmodel == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.train_batch_size)
    # for batch_idx, batch in enumerate(train_dataset):
    #     input = batch['net_input']['src_tokens'].to(args.device).t()
    #     # input_lengths = batch['net_input']['src_lengths'].to(args.device)
    #     targets = batch['target'].to(args.device).t()
    batch_idx, i = 0, 0
    while i < train_dataset.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        input, targets = get_batch(train_dataset, i, args, seq_len=seq_len)
        model.train()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden, rnn_hs, dropped_rnn_hs = model(input, hidden, return_h=True)
        #rnn_hs is [layer_size, lengths, batch_size, hidden_state_size]
       
        # output, hidden = model(data, hidden, return_h=False)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        loss = raw_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            try:
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch_idx, len(train_dataset) // args.bptt, optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            except OverflowError:
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl Inf | bpc {:8.3f}'.format(
                    epoch, batch_idx, len(train_dataset) // args.bptt, optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        batch_idx += 1
        i += seq_len
        

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../data/news/',
                        help='location of the data corpus')
    parser.add_argument('--dependency_path', type=str, default='../data/news/dependency',
                        help='location of the data dependency ')
    parser.add_argument('--dataname', type=str, default='news',
                        help='name of the data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (AWD-LSTM, ONLSTM)')
    parser.add_argument('--rnnmodel', type=str, default='LSTM',
                        help='type of recurrent cell (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='number of units per chunk')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--train_batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument("--per_gpu_train_batch_size", default=96, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.25,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.5,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.4,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--do_eval', action='store_true',
                        help='do evaluate')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
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
    parser.add_argument('--trained_epoches', type=int, default=0,
                        help='have trained epoches')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--finetuning', type=int, default=500,
                        help='When (which epochs) to switch to finetuning')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument('--philly', action='store_true',
                        help='Use philly cluster')
    args = parser.parse_args()
    args.tied = True
    return args

def main():
    args = get_args()
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    # device = torch.device("cpu")
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    

    ###############################################################################
    # Load data
    ###############################################################################

    fn = 'data/corpus.{}.data'.format(args.dataname)
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    if os.path.exists(fn):
        logger.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logger.info('Producing dataset...')
        corpus = data.DependencyCorpus(args.data, args.dependency_path, "train.head", "valid.head", "test.head")
        torch.save(corpus, fn)
    
    # train_dataset = data.SentenceDataset(corpus.train, corpus.train_lengths, corpus.dictionary)
    # valid_dataset = data.SentenceDataset(corpus.valid, corpus.valid_lengths, corpus.dictionary)
    # test_dataset = data.SentenceDataset(corpus.test, corpus.test_lengths, corpus.dictionary)
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # eval_batch_size = 10
    # test_batch_size = 1

    # collate_func_for_use = partial(collate_func_for_tok, 
    #     pad_idx=corpus.dictionary.word2idx['<pad>'],
    #     eos_idx=corpus.dictionary.word2idx['<eos>'])
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=5, shuffle=False, drop_last=True, collate_fn=collate_func_for_use)

    # eval_sampler = SequentialSampler(valid_dataset) if args.local_rank == -1 else DistributedSampler(valid_dataset)
    # valid_loader = DataLoader(valid_dataset, sampler=eval_sampler,  batch_size=eval_batch_size, num_workers=5, shuffle=False, drop_last=True, collate_fn=collate_func_for_use)
    # test_loader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=test_batch_size, num_workers=5, shuffle=False, drop_last=True, collate_fn=collate_func_for_use)
    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train_allinone, args.train_batch_size, args)
    val_data = batchify(corpus.valid_allinone, eval_batch_size, args)
    test_data = batchify(corpus.test_allinone, test_batch_size, args)



    from splitcross import SplitCrossEntropyLoss
    criterion = None
    ntokens = len(corpus.dictionary)
    
    if args.model == "ONLSTM":
        model = ONLSTMModel(args.rnnmodel, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                            args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    elif args.model == "LSTM":
        model = RNNModel(args.rnnmodel, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    else:
        raise ValueError(" model must be ONLSTM or LSTM")

    ###############################################################################
    # Build the model
    ###############################################################################
    ###
    if args.resume:
        logger.info('Resuming model ...')
        model, criterion, optimizer = model_load(args, args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            if args.model == "ONLSTM":
                for rnn in model.rnn.cells:
                    rnn.hh.dropout = args.wdrop
            elif args.model == "LSTM":
                from weight_drop import WeightDrop
                for rnn in model.rnns:
                    if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                    elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
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
        logger.info('Using {}'.format(splits))
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    ###
    if args.cuda:
        model = model.to(args.device)
        criterion = criterion.to(args.device)

    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    logger.info('Args: {}'.format(args))
    logger.info('Model total parameters: {}'.format(total_params))

    

    # At any point you can hit Ctrl + C to break out of training early.
    if not args.do_eval:
        # Loop over epochs.
        lr = args.lr
        best_val_loss = []
        stored_loss = 100000000
        try:
            optimizer = None
            # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
            # Train!
        
            for epoch in range(args.trained_epoches + 1, args.epochs + 1):
                epoch_start_time = time.time()
                train_a_epoch(args, train_data, model, corpus, optimizer, criterion, params, epoch)
                if 't0' in optimizer.param_groups[0]:
                    tmp = {}
                    for prm in model.parameters():
                        tmp[prm] = prm.data.clone()
                        if 'ax' in optimizer.state[prm]:
                            prm.data = optimizer.state[prm]['ax'].clone()

                    val_loss2 = evaluate(args, val_data, model, corpus, criterion, eval_batch_size)
                    logger.info('-' * 89)
                    try:
                        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                    except OverflowError:
                        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl inf | valid bpc {:8.3f}'.format(
                            epoch, (time.time() - epoch_start_time), val_loss2, val_loss2 / math.log(2)))
                    logger.info('-' * 89)

                    if val_loss2 < stored_loss:
                        model_save(args, args.save, model, criterion, optimizer)
                        logger.info('Saving Averaged!')
                        stored_loss = val_loss2

                    for prm in model.parameters():
                        prm.data = tmp[prm].clone()

                    if epoch == args.finetuning:
                        logger.info('Switching to finetuning')
                        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                        best_val_loss = []

                    if epoch > args.finetuning and len(best_val_loss) > args.nonmono and val_loss2 > min(
                            best_val_loss[:-args.nonmono]):
                        logger.info('Done!')
                        import sys

                        sys.exit(1)

                else:
                    val_loss = evaluate(args, val_data, model, corpus, criterion, eval_batch_size)
                    logger.info('-' * 89)
                    try:
                        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                    except OverflowError:
                        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl Inf | valid bpc {:8.3f}'.format(
                            epoch, (time.time() - epoch_start_time), val_loss, val_loss / math.log(2)))
                    logger.info('-' * 89)

                    if val_loss < stored_loss:
                        model_save(args, args.save, model, criterion, optimizer)
                        logger.info('Saving model (new best validation)')
                        stored_loss = val_loss

                    if args.optimizer == 'adam':
                        scheduler.step(val_loss)

                    if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                            len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                        logger.info('Switching to ASGD')
                        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                    if epoch in args.when:
                        logger.info('Saving model before learning rate decreased')
                        model_save(args, '{}.e{}'.format(args.save, epoch), model, criterion, optimizer)
                        logger.info('Dividing learning rate by 10')
                        optimizer.param_groups[0]['lr'] /= 10.

                    best_val_loss.append(val_loss)

                logger.info("PROGRESS: {}%".format((epoch / args.epochs) * 100))

        except KeyboardInterrupt:
            logger.info('-' * 89)
            logger.info('Exiting from training early')
    else:
        logger.info('| Begin of testing |')
        # Load the best saved model.
        model, criterion, optimizer = model_load(args, args.save)

        # Run on test data.
        test_loss =  evaluate(args, test_data, model, corpus, criterion, test_batch_size)
        logger.info('=' * 89)
        try:
            logger.info('| End of testing | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
                test_loss, math.exp(test_loss), test_loss / math.log(2)))
        except OverflowError:
            logger.info('| End of testing | test loss {:5.2f} | test ppl Inf | test bpc {:8.3f}'.format(
                test_loss, test_loss / math.log(2)))
        logger.info('=' * 89)

if __name__ == '__main__':
    main()