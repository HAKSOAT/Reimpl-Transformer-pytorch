from torch import nn
import torch

from utils.constants import SPECIAL_TOKENS_INDEX, PAD_TOKEN


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocabulary_size,
                 pad_index=SPECIAL_TOKENS_INDEX[PAD_TOKEN]):
        assert 0.0 < label_smoothing <= 1.0

        super(LabelSmoothingLoss, self).__init__()

        self.pad_index = pad_index
        # Q: Why is LogSoftmax being used?
        # A: This LogSoftmax is being used because the criterion being used here,
        # KLDivLoss expects log probabilities:
        # https://stackoverflow.com/questions/62806681/pytorch-kldivloss-loss-is-negative
        # https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
        # Doing logsoftmax implies that the distribution of the probabilities won't be [0, 1] and won't sum to 1.
        # Instead it will be [-inf, 0]. Somewhere in KLDivLoss or for anybody using the results of logsoftmax
        # They can easily get back the softmax probabilities by doing an exp of the probabilities,
        # something like np.exp(probabilities). This aside, log-softmax is said to be more stable for training purposes.
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction="sum")

        smoothing_value = label_smoothing / (vocabulary_size - 2)
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_index] = 0
        self.register_buffer("smoothed_targets", smoothed_targets.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()
        outputs_log_sotmax = self.log_softmax(outputs)
        outputs_flat = outputs_log_sotmax.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size, seq_len)

        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)
        smoothed_targets.masked_fill_((targets_flat == self.pad_index).unsqueeze(1), 0)

        loss = self.criterion(outputs_flat, smoothed_targets)
        count = (targets != self.pad_index).sum().item()

        return loss, count


class TokenCrossEntropyLoss:
    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)
        count = (targets != self.pad_index).sum().item()

        return batch_loss, count

