import math
import torch
from dataclasses import dataclass
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch.nn import BCEWithLogitsLoss
import time

@register_criterion("dependency_cross_entropy")
class DependencyCrossEntropy(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample["target"]
        target = target.view(-1, target.size(-1))
        # loss = 0
        loss = torch.zeros(lprobs.size(0)).to(lprobs.device)
        idx_range = torch.arange(target.size(-1)).to(lprobs.device)
        
        for i in range(lprobs.size(0)):
            start_time = time.time()
            dependency_num = torch.sum(target[i]).item()
            dependency_num_time = time.time()
            # print("dependency_num_time is : ", dependency_num_time - start_time)
            if dependency_num > 0:
                cur_logit = lprobs[i].unsqueeze(0)
                cur_logit = cur_logit.expand(int(dependency_num), cur_logit.size(-1))
                cur_logit_time = time.time()
                # print("cur_logit_time is : ", cur_logit_time - dependency_num_time)
                cur_target = idx_range.masked_select(target[i].bool())
                cur_target_time = time.time()
                # print("cur_target_time is : ", cur_target_time - cur_logit_time)
                loss[i] = F.nll_loss(
                    cur_logit,
                    cur_target,
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )
                nll_loss_time = time.time()
                # print("nll_loss_time is : ", nll_loss_time - cur_target_time)
        # print("return loss")
        return torch.sum(loss)
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True