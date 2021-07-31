import argparse
from model import RNNModel
from main import model_load
from utils import repackage_hidden
import torch
import torchtext
from torch.autograd import Variable
import torch.nn.functional as F
import data
import os
import math
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--dataname', type=str, default='news',
                        help='name of the data')
parser.add_argument('--dependency_path', type=str, default='../data/news/dependency',
                        help='location of the data dependency ')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--is_dp_model', action='store_true',
                    help='Use dependency pointer model')
parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')  
parser.add_argument('--context_length', type=int, default=36,
                    help='context to conduct dependency pointer network')              
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--task', type=str, default='unconditional_gen', choices=['unconditional_gen', 'story_gen'])
parser.add_argument('--story_input', type=str, default='../data/rocstories/test_input.txt')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--topk', type=int, default=0,
                    help='topk - higher will increase diversity')
parser.add_argument('--topp', type=float, default=0.0,
                    help='topp - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--philly', action='store_true',
                    help='Use philly cluster')
args = parser.parse_args()
args.tied = True
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

logger.info('Generating with top-p {}, top-k {}...'.format(args.topp, args.topk))

fn = 'data/corpus.{}.data'.format(args.dataname)
if args.philly:
    fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
if os.path.exists(fn):
    logger.info('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    logger.info('Producing dataset...')
    corpus = data.DependencyCorpus(args.data, args.dependency_path, "train.head", "valid.head", "test.head")
ntokens = len(corpus.dictionary)
with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        model = torch.load(f, map_location='cpu')

model = model[0]
model.eval()
if args.model == 'QRNN':
    model.reset()

model.to(device)

if args.task == 'unconditional_gen':
    hidden = model.init_hidden(1)
    input = torch.LongTensor([[corpus.dictionary.word2idx['<eos>']]]).to(device)
    eos_num = 0
    with open(args.outf, 'w') as outf:
        if args.is_dp_model:
            raw_output_history = torch.zeros((0, 1, args.emsize), dtype=torch.float).to(device)
            dep_logits_history = torch.zeros((1, 0, ntokens), dtype=torch.float).to(device)
            for i in range(args.words):
                raw_output, hidden = model.generate_step(input, hidden)
                # raw_output: [1, 1, embeding_size]
                raw_output_history = torch.cat((raw_output_history, raw_output), dim=0)
                # raw_output_history: [seq_len, 1, embeding_size]
                query = model.query(raw_output).transpose(0,1)
                attention  = torch.bmm(query, raw_output_history.permute(1,2,0))
                attention = attention / math.sqrt(raw_output.size(-1)) #[1, 1, seq_len]
                if args.context_length >= 0:
                    attention[0][0][:-args.context_length] = -float('Inf')
                weight = F.softmax(attention, dim=-1)
                dep_logit = model.decoder(model.lockdrop(raw_output, model.dropout)).transpose(0,1) # [1(batch), 1(seqlen), ntokens]
                dep_logits_history = torch.cat((dep_logits_history, dep_logit), dim=1)
                word_weights = torch.bmm(weight, dep_logits_history).squeeze().data.div(args.temperature)
                word_weights = top_k_top_p_filtering(word_weights, args.topk, args.topp)
                word_idx = torch.multinomial(torch.softmax(word_weights,-1), 1).cpu()[0]
                input.fill_(word_idx)
                word = corpus.dictionary.idx2word[word_idx]

                # hidden = model.init_hidden(1)
                # dep_logits, attn_weight, hidden = model(input, hidden, return_h=False, context_length=args.context_length)
                # dep_logits = dep_logits.transpose(0,1)
                # # dep_logits is [batch_size, seq_len, ntokens]
                # output = torch.bmm(attn_weight, dep_logits).transpose(0,1)
                # word_weights = output[-1].squeeze().data.div(args.temperature)
                # word_weights = top_k_top_p_filtering(word_weights, args.topk, args.topp)
                # word_idx = torch.multinomial(torch.softmax(word_weights,-1), 1).cpu()[0]
                # input = torch.cat((input,torch.LongTensor([[word_idx]]).to(device)), dim=0)
                # word = corpus.dictionary.idx2word[word_idx]

                if word == '<eos>':
                    outf.write('\n')
                    eos_num += 1
                    if eos_num == 1000:
                        break
                else:
                    outf.write(word +  ' ')
                hidden = repackage_hidden(hidden)
                    
                if i % args.log_interval == 0:
                    logger.info('| Generated {}/{} words'.format(i, args.words))

        else:
            for i in range(args.words):
                # print("input is : ", input)
                output, hidden = model(input, hidden)
                # print("output size is : ", output.size())
                word_weights = model.decoder(output).squeeze().data.div(args.temperature)
                word_weights = top_k_top_p_filtering(word_weights, args.topk, args.topp)
                word_idx = torch.multinomial(torch.softmax(word_weights,-1), 1).cpu()[0]
                input.fill_(word_idx)
                word = corpus.dictionary.idx2word[word_idx]
                if word == '<eos>':
                    outf.write('\n')
                    eos_num += 1
                    if eos_num == 1000:
                        break
                else:
                    outf.write(word +  ' ')
                hidden = repackage_hidden(hidden)
                if i % args.log_interval == 0:
                    logger.info('| Generated {}/{} words'.format(i, args.words))
elif args.task == "story_gen":
    outf = open(args.outf, 'w')
    story_input = open(args.story_input, 'r')
    story_input_lines = story_input.readlines()
    story_input.close()
    total_input_lines_num = len(story_input_lines)
    for i, line in enumerate(story_input_lines):
        line_split = ['<eos>'] + line.strip().split()

        hidden = model.init_hidden(1)
        input = torch.LongTensor([[corpus.dictionary.word2idx[token] for token in line_split]]).to(device)
        input = input.transpose(0,1)
        for _ in range(1000):
            if args.is_dp_model:
                if i % args.log_interval == 0:
                    logger.info('| Generated {}/{}'.format(i, total_input_lines_num))
            else:
                output, hidden = model(input, hidden)
                output = output[-1]
                word_weights = model.decoder(output).data.div(args.temperature)
                word_weights = top_k_top_p_filtering(word_weights, args.topk, args.topp)
                word_idx = torch.multinomial(torch.softmax(word_weights,-1), 1).cpu()[0]
                input = torch.LongTensor([[word_idx]]).to(device)
                word = corpus.dictionary.idx2word[word_idx]
                if word == '<eos>':
                    outf.write('\n')
                    break
                else:
                    outf.write(word +  ' ')
                hidden = repackage_hidden(hidden)
        if i % args.log_interval == 0:
            logger.info('| Generated {}/{}'.format(i, total_input_lines_num))

else:
    raise ValueError("The task must be unconditional_gen or story_gen")