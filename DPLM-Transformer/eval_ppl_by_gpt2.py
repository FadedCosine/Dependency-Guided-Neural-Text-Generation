from    tqdm import tqdm
from    transformers import GPT2LMHeadModel, GPT2TokenizerFast
import  argparse
import  os
import torch
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        if 'iternums' not in path:
            paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                if 'iternums' not in fname and fname.endswith('.txt'):
                    paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)
    return paths
@torch.no_grad()
def cal_ppl(filenames, model_file):
    ppl_dir = {}
    for filename in filenames:
        with open(filename, 'r') as f:
            sents = []
            for line in f.readlines():
                line_split = line.strip('\n').strip().lower()
                if len(line_split) != 0:
                    sents.append(line_split)
        model = GPT2LMHeadModel.from_pretrained(model_file).cuda()
        tokenizer = GPT2TokenizerFast.from_pretrained(model_file)
        model.eval()
        ppl = torch.FloatTensor(len(sents)).cuda()
        max_length = model.config.n_positions
        for index, sent in tqdm(enumerate(sents)):
            encodings = tokenizer(sent, return_tensors='pt')
            input_ids = encodings.input_ids[:, :max_length].cuda()
            target_ids = input_ids.clone()
            outputs = model(input_ids, labels=target_ids)
            ppl[index] = torch.exp(outputs[0]).tolist()
        ppl_dir[filename] = torch.mean(ppl)
    for i in ppl_dir.keys():
        logger.info('{:<65}{:.3f}'.format(os.path.basename(i), ppl_dir[i]))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-p',type=float, default=0.9)
    parser.add_argument('--top-k',type=int, default=0)
    parser.add_argument('--model_file', type=str, default="/home/yangzhixian/pretrained_model/gpt2-base")
    parser.add_argument('--cuda_num', type=str, default='3')
    parser.add_argument('--folderpath', type=str, default="checkpoints/ptb/samples/")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    # folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}".format(p=args.top_p, k=args.top_k))
    folderpath = args.folderpath
    # print("=" * 20 + "topp-{p}-topk-{k}-temp-1".format(p=args.top_p, k=args.top_k) + "=" * 20)
    filenames = sorted(get_files(folderpath))
    logger.info(filenames)
    cal_ppl(filenames, args.model_file)