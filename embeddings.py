import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    # Q: What does max_len here mean?
    # Q: What does padding_idx here mean?
    # A: When token ids are converted into embeddings
    # nn.Embeddings will assign vector in the embedding to the corresponding ids
    # so by setting padding_idx, ids that match that idx will have a vector of zero.
    # Q: Why are we using dropout_prob here?
    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0., padding_idx=0, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                              -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embedding.weight
        # Q: I think this is a really nice trick to save the positional Encodings pe.
        # Learnt something new about the use of register_buffer here.
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embedding(x)
        # Q: Why did they multiply by this here? It is not in the paper.
        x = x * math.sqrt(self.dim)
        # My understanding of this step check is that if step is not provided, then
        # we add all the positional encodings to the input vector.
        # Otherwise, we skip as provided in the step.
        if step is None:
            # Q: I find it very interesting to see that pe can be accessed as an instance variable here
            # simply because we registered buffer. Awesome stuffff.
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x
