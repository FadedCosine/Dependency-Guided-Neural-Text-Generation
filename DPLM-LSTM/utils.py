import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.to(args.device)
    return data
def batchify_dep_tokenlist(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.

    data = data.reshape(bsz, -1).T
    return data
def batchify_dep_indicators(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.

    nbatch = data.size(0) // bsz
    ntokens = data.size(-1)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1, ntokens).transpose(0,1).contiguous()
    if args.cuda:
        data = data.to(args.device)
    return data

def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
    
def get_dep_batch(source, source_dependency, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source_dependency[i:i+seq_len]
    return data, target

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

"""
collate function，从dataloader给出的batch中，依据batch中的item中最大的长度，做padding
"""
def collate_func_for_dep(samples, pad_idx, eos_idx):
    batch_len=len(samples)
    if batch_len == 0:
        return {}
    def merge(key):
        if key == "source":
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )
        elif key == "target":
            values = [s[key] for s in samples]
            vocab_size = values[0].size(-1)
            size = max(v.size(0) for v in values)
            for i in range(len(values)):
                values[i] = torch.cat([values[i], torch.zeros(size - values[i].size(0), vocab_size)], dim=0)
            return torch.stack(values)

    #src_tokens即为batch中的sources，依据这个batch最长的sources经过padding之后得到的2d tensor
    src_tokens = merge("source")
  
    #此时会是samples[0]["target"]是tensor,[seq_len, vocab_size]
    target = merge("target")
    #返回的target是[batch_size, seq_batch_max_len, vocab_size]
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
        },
        "target": target,
        "seq_true_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
    }

def collate_func_for_tok(samples, pad_idx, eos_idx):
    batch_len=len(samples)
    if batch_len == 0:
        return {}
    #src_tokens即为batch中的sources，依据这个batch最长的sources经过padding之后得到的2d tensor
    src_tokens = collate_tokens(
                [s["source"] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )
    
    target = collate_tokens(
                [s["target"] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )
    #返回的target是[batch_size, seq_batch_max_len, vocab_size]
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
        },
        "target": target,
        "seq_true_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
    }

def load_embeddings_txt(path):
  words = pd.read_csv(path, sep=" ", index_col=0,
                      na_values=None, keep_default_na=False, header=None,
                      quoting=csv.QUOTE_NONE)
  matrix = words.values
  index_to_word = list(words.index)
  word_to_index = {
    word: ind for ind, word in enumerate(index_to_word)
  }
  return matrix, word_to_index, index_to_word

def evalb(pred_tree_list, targ_tree_list):
    import os
    import subprocess
    import tempfile
    import re
    import nltk

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    print("Temp: {}, {}".format(temp_file_path, temp_targ_path))
    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    for pred_tree, targ_tree in zip(pred_tree_list, targ_tree_list):
        def process_str_tree(str_tree):
            return re.sub('[ |\n]+', ' ', str_tree)

        def list2tree(node):
            if isinstance(node, list):
                tree = []
                for child in node:
                    tree.append(list2tree(child))
                return nltk.Tree('<unk>', tree)
            elif isinstance(node, str):
                return nltk.Tree('<word>', [node])

        temp_tree_file.write(process_str_tree(str(list2tree(pred_tree)).lower()) + '\n')
        temp_targ_file.write(process_str_tree(str(list2tree(targ_tree)).lower()) + '\n')

    temp_tree_file.close()
    temp_targ_file.close()

    evalb_dir = os.path.join(os.getcwd(), "EVALB")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    subprocess.run(command, shell=True)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_fscore = float(match.group(1))
                break

    temp_path.cleanup()

    print('-' * 80)
    print('Evalb Prec:', evalb_precision,
          ', Evalb Reca:', evalb_recall,
          ', Evalb F1:', evalb_fscore)

    return evalb_fscore
