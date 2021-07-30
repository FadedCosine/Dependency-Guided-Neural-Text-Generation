import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn

class DependencyCrossEntropy(_Loss):
    def __init__(self):
        super(DependencyCrossEntropy, self).__init__()
    def forward(self, logits, target):
        # target : [seq_len, batch_size, {dep_token_set}]
        seq_len, batch_size = target.shape
        target_indicators = torch.zeros((seq_len * batch_size, logits.size(-1))).to(logits.device)
        for idx, dep_set in enumerate(target.reshape(-1)):
                target_indicators[idx, list(dep_set)] = 1
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = torch.sum(torch.bmm(torch.neg(lprobs).unsqueeze(1), target_indicators.unsqueeze(-1))) / len(target_indicators) 
        # print("loss is : ", loss)
        return loss
