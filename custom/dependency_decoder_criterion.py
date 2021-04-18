import math
from dataclasses import dataclass
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch.nn import BCEWithLogitsLoss

@register_criterion("dependency_prediction")
class DependencyBCE(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss = self.compute_loss(model, net_output, sample)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        preds = torch.round(F.sigmoid(net_output[0])).int()
        T = torch.logical_and(preds, sample["target"].int())
        P = torch.sum(T) / torch.sum(preds)
        R = torch.sum(T) / torch.sum(sample["target"].int())
        logging_output["ncorrect"] = torch.sum(T) 
        logging_output["target_ones"] = torch.sum(sample["target"].int())
        logging_output["predict_ones"] = torch.sum(preds)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        logits = net_output[0]
        # logits = logits.view(-1, logits.size(-1)) 
        target = sample["target"]
        lengths = sample["seq_true_lengths"]
        padding_mask = torch.ones(target.size())
        for idx, seq_len in enumerate(lengths):
            padding_mask[idx, seq_len:, :] = 0
        # target = target.view(-1, target.size(-1)) 
        # padding_mask = padding_mask.view(-1, padding_mask.size(-1)) 
        assert logits.size() == target.size(), "model output logits's size must be consist with target's size!"
        criterion = BCEWithLogitsLoss(reduction="none")
        return torch.sum(criterion(logits, target) * padding_mask)

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
        ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
        target_ones = sum(log.get("target_ones", 0) for log in logging_outputs)
        predict_ones = sum(log.get("predict_ones", 0) for log in logging_outputs)
        metrics.log_scalar(
            "precision", 100.0 * ncorrect / predict_ones, predict_ones, round=1
        )
        metrics.log_scalar(
            "recall", 100.0 * ncorrect / target_ones, target_ones, round=1
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True