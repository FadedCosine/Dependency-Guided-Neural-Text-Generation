import numpy as np
import torch
import torch.utils.data
import argparse
import os
from metrics import *
import logging
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s')
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

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-p',type=float, default=0.9)
    parser.add_argument('--top-k',type=int, default=0)
    parser.add_argument('--folderpath',type=str, default="checkpoints/rocstories/samples/")
    parser.add_argument('--data_dir',type=str, default="../data/rocstories")
    return parser.parse_args()
def main():
    args = get_args()
    # folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}".format(p=args.top_p, k=args.top_k))
    folderpath = args.folderpath
    filenames = sorted(get_files(folderpath))
    logger.info(filenames)
    bleu = {}
    self_bleu = {}
    dist = {}
    uniq = {}
    repit = {}
    ground_truth = open(os.path.join(args.data_dir,'test_output.txt'), 'r').readlines()
    ground_truth = [line.strip().split()for line in ground_truth]
    for filename in filenames:
        logger.info('evaluating {}...'.format(filename))
        pred_org = open(filename, 'r').readlines()
        predict = []
        gt = []
        for line_idx in range(len(pred_org)):
            line_split = pred_org[line_idx].strip().split()
            if len(line_split) != 0:
                predict.append(line_split)
                gt.append(ground_truth[line_idx])
        # bleu[filename] = bleu_upto(gt, predict, 5)
        # self_bleu[filename] = selfbleu(predict, 5)
        dist[filename] = distinct_document(predict, 4)
        # repit[filename] = repetition(predict)
        # s = set()
        # for i in predict:
        #     s.update(i)
        # uniq[filename] = len(s)
    
  
    # logger.info('--------------------repetition(Down)----------------------')
    # for i in bleu.keys():
    #     logger.info('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))

    # logger.info('--------------------bleu(Up)----------------------')
    # for i in bleu.keys():
    #     logger.info('{:<65}{}, {}, {}, {}, {}'.format(os.path.basename(i), round(bleu[i][0] * 100, 2), round(bleu[i][1] * 100,2), round(bleu[i][2] * 100,2), round(bleu[i][3] * 100,2), round(bleu[i][4] * 100,2)))
    # logger.info('--------------------self-bleu(Down)----------------------')
    # for i in self_bleu.keys():
    #     logger.info('{:<65}{}, {}, {}, {}, {}'.format(os.path.basename(i), round(self_bleu[i][0] * 100, 2), round(self_bleu[i][1] * 100,2), round(self_bleu[i][2] * 100,2), round(self_bleu[i][3] * 100,2), round(self_bleu[i][4] * 100,2)))

    logger.info('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        logger.info('{:<64}{}, {}, {}, {}'.format(os.path.basename(i), round(dist[i][0] * 100,2), round(dist[i][1] * 100,2), round(dist[i][2] * 100,2), round(dist[i][3] * 100,2)))
    # logger.info('--------------------uniq_seq(Up)----------------------')
    # for i in uniq.keys():
    #     logger.info('{:<64}{}'.format(os.path.basename(i), uniq[i]))


if __name__ =='__main__':
    main()
