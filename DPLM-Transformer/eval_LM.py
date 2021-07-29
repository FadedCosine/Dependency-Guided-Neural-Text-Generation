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
    parser.add_argument('--folderpath',type=str, default="checkpoints/new/samples/")
    parser.add_argument('--data_dir',type=str, default="../data/news")
    return parser.parse_args()
def main():
    args = get_args()
    # folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}".format(p=args.top_p, k=args.top_k))
    folderpath = args.folderpath
    filenames = sorted(get_files(folderpath))
    logger.info(filenames)
    kd = {}
    msj = {}
    bleu = {}
    self_bleu5 = {}
    dist = {}
    uniq = {}
    gt = open(os.path.join(args.data_dir,'test.txt'), 'r').readlines()
    gt = [line.strip().split()for line in gt]
    for filename in filenames:
        predict = open(filename, 'r').readlines()
        predict = [line.strip().split()for line in predict]  
        kd[filename] = kld(gt, predict, 1)
        msj[filename] = ms_jaccard(gt, predict, 5)
        bleu[filename] = Bleu(test_text=filename,
                real_text=os.path.join(args.data_dir,'test.txt'),
                num_real_sentences=10000,
                num_fake_sentences=10000,
                gram=5).get_score()
        
        self_bleu5[filename] = SelfBleu(test_text=filename,
                        num_sentences=10000,
                        gram=5).get_score()
        dist[filename] = distinct_upto(predict, 5)
        s = set()
        for i in predict:
            s.update(i)
        uniq[filename] = len(s)
    
    logger.info('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        logger.info('{:<64}{}'.format(os.path.basename(i), round(kd[i],2)))

    logger.info('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        logger.info('{:<64}{}, {}, {}'.format(os.path.basename(i),  round(msj[i][0] * 100,1),  round(msj[i][1] * 100,1),  round(msj[i][2] * 100,1)))

    logger.info('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        logger.info('{:<64}{}, {}, {}'.format(os.path.basename(i), round(dist[i][0] * 100,1), round(dist[i][1] * 100,1), round(dist[i][2] * 100,1)))

    logger.info('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        logger.info('{:<64}{}'.format(os.path.basename(i), uniq[i]))

    logger.info('--------------------bleu(Up)----------------------')
    for i in bleu.keys():
        logger.info('{:<64}{}'.format(os.path.basename(i), round(bleu[i]*100,1)))
        
    logger.info('--------------------self-bleu(Down)----------------------')
    for i in self_bleu5.keys():
        logger.info('{:<64}{}'.format(os.path.basename(i), round(self_bleu5[i]*100,1)))



if __name__ =='__main__':
    main()
