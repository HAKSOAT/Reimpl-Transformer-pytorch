import numpy as np
import torch

from utils.constants import SPECIAL_TOKENS_INDEX, PAD_TOKEN


def pad_masking(x, target_len):
    batch_size, seq_len = x.size()
    padded_positions = x == SPECIAL_TOKENS_INDEX[PAD_TOKEN]
    # Q: Look at the data values in here to understand what this expand does in this scenario.
    # Q: How is expand different from repeat?
    # A: expand() will never allocate new memory. And so require the expanded dimension to be of size 1.
    # repeat() will always allocate new memory and the repeated dimension can be of any size.
    pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
    return pad_mask


def subsequent_masking(x):
    batch_size, seq_len = x.size()
    # Q: Why is np.triu being used instead of tril? If indeed this is for the masked multi-head,
    # then shouldn't we be keeping the lower matrix?
    subsequent_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype("uint8")
    subsequent_mask = torch.tensor(subsequent_mask).to(x.device)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subsequent_mask