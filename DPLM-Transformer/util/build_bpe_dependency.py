import argparse
import os
import logging
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def build_word_to_bpe_index_list(tok_sentences, bpe_sentences):
    """
    token sentence:   copepods          are small          crustaceans .
    bpe sentence:     cop@@ e@@ po@@ ds are small cru@@ stac@@ e@@ ans .
    return: [[0,1,2,3], [4], [5], [6, 7, 8, 9], [10]]
    """
    word_to_bpe_index_list = []
    for tok_sentence, bpe_sentence in zip(tok_sentences, bpe_sentences):
        tok_sentence = tok_sentence.strip().split()
        bpe_sentence = bpe_sentence.strip().split()
        
        cur_word_to_bpe_index = []
        pre_bpe_butter = []
        for bpe_idx, bpe_sw in enumerate(bpe_sentence):
            if (len(pre_bpe_butter) == 0 and '@' not in bpe_sw) or (bpe_sw =='@'):
                cur_word_to_bpe_index.append([bpe_idx])
            elif '@' in bpe_sw:
                pre_bpe_butter.append(bpe_idx)
            else: # the last subword
                pre_bpe_butter.append(bpe_idx)
                cur_word_to_bpe_index.append(pre_bpe_butter)
                pre_bpe_butter = []
        if len(cur_word_to_bpe_index) != len(tok_sentence):
            print("tok_sentence is : ", " ".join(tok_sentence))
            print("bpe_sentence is : ", " ".join(bpe_sentence))
            print("cur_word_to_bpe_index is : ", cur_word_to_bpe_index)
        assert len(cur_word_to_bpe_index) == len(tok_sentence)
        word_to_bpe_index_list.append(cur_word_to_bpe_index)
    return word_to_bpe_index_list

def build_subword_dependecy_head(tok_dependency_headlist, word_to_bpe_index_list):
    subword_dependecy_head = []
    for tok_dep_head, word_to_bpe_index in zip(tok_dependency_headlist, word_to_bpe_index_list):
        cur_bpe_dep_head = []
        for tok_head, word_to_bpe in zip(tok_dep_head, word_to_bpe_index):
            # the head of the old word is now the head of the first subword; every dependency of the old word node now depends on the last subword.
            if tok_head-1 >= 0:
                cur_bpe_dep_head.append(word_to_bpe_index[tok_head-1][-1] + 1)
            else:
                cur_bpe_dep_head.append(0)
            # each subsequent subword depends on the previous one
            for res_bpe in word_to_bpe[1:]:
                cur_bpe_dep_head.append(res_bpe)
        subword_dependecy_head.append(cur_bpe_dep_head)
    return subword_dependecy_head

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_level_path",type=str, default="/home/yangzhixian/DependencyGuided/DPDecoder/data/news/tmp")
    parser.add_argument('--filename',type=str, default="valid.txt")
    parser.add_argument("--bpe_level_path",type=str, default="/home/yangzhixian/DependencyGuided/DPDecoder/data/news")
    parser.add_argument('--dependency_filename',type=str, default="valid.head")

    return parser.parse_args()
def main():
    args = get_args()
    with open(os.path.join(args.token_level_path, args.filename), 'r') as f:
        tok_sentences = f.readlines()
    with open(os.path.join(args.bpe_level_path, args.filename), 'r') as f:
        bpe_sentences = f.readlines()
    word_to_bpe_index_list = build_word_to_bpe_index_list(tok_sentences, bpe_sentences)

    tok_dependency_headlist = []
    with open(os.path.join(args.token_level_path, "dependency", args.dependency_filename), "r", encoding="utf-8") as f:
        for line in f:
            tok_dependency_headlist.append(eval(line.strip("\n")))
    subword_dependecy_head = build_subword_dependecy_head(tok_dependency_headlist, word_to_bpe_index_list)
    for sw_dependecy_head, bpe_sentence in zip(subword_dependecy_head, bpe_sentences):
        bpe_sentence = bpe_sentence.strip().split(' ')
        if len(sw_dependecy_head) != len(bpe_sentence):
            print("bpe_sentence join is : ", " ".join(bpe_sentence))
            print("sw_dependecy_head is : ", sw_dependecy_head)
        assert len(bpe_sentence) == len(sw_dependecy_head)
    with open(os.path.join(args.bpe_level_path, "dependency", args.dependency_filename), "w", encoding="utf-8") as f:
        for dependecy_head in subword_dependecy_head:
            f.write(str(dependecy_head)+'\n')


if __name__ =='__main__':
    main()
