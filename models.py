from collections import defaultdict

import numpy as np
import torch
from torch import nn

from embeddings import PositionalEncoding
from pad import pad_masking, subsequent_masking


def build_model(config, source_vocabulary_size, target_vocabulary_size):
    if config["positional_encoding"]:
        source_embedding = PositionalEncoding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=config["d_model"],
            dim=config["d_model"]
        )
        target_embedding = PositionalEncoding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=config["d_model"],
            dim=config["d_model"]
        )
    else:
        source_embedding = nn.Embedding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=config["d_model"]
        )
        target_embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=config["d_model"]
        )

    encoder = TransformerEncoder(
        layers_count=config["layers_count"],
        d_model=config["d_model"],
        heads_count=config["heads_count"],
        d_ff=config["d_ff"],
        dropout_prob=config["dropout_prob"],
        embedding=source_embedding
    )

    decoder = TransformerDecoder(
        layers_count=config["layers_count"],
        d_model=config["d_model"],
        heads_count=config["heads_count"],
        d_ff=config["d_ff"],
        dropout_prob=config["dropout_prob"],
        embedding=target_embedding
    )

    model = Transformer(encoder, decoder)

    return model


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, inputs):
        # The batch size here is referring to the number of documents
        # While sources_len is referring to the size of each document
        batch_size, sources_len = sources.size()
        batch_size, inputs_len = inputs.size()

        sources_mask = pad_masking(sources, sources_len)
        # Q: Why is sources and inputs_len used here instead of inputs and inputs len?
        # A: Because memory mask is masking the encoder's embeddings which uses sources.
        # Q: During inference, how is this inputs_len useful?
        memory_mask = pad_masking(sources, inputs_len)
        # Q: What does subsequent_masking do?
        # A: It enforces the masked language modelling approach, ensuring that only the words
        # that come before the word to be predicted are already seen and that the words after it are never
        # seen so that the model does not cheat.
        inputs_mask = subsequent_masking(inputs) | pad_masking(inputs, inputs_len)

        memory = self.encoder(sources, sources_mask)
        # Q: Since we are not passing state or layer_cache at this point,
        # I am concerned that it will never be used. So I need to run this code to check this out.
        # A: The concern is right, the cache is never used during the training process. However, it is used
        # during predictions. Truly, it does not make sense to cache during training as the weights will be
        # updated during backprop, making the stored weights irrelevant if cached.
        outputs, state = self.decoder(inputs, memory, memory_mask, inputs_mask)
        return outputs


class TransformerEncoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerEncoder, self).__init__()

        # Q: What purpose does d_model serve?
        # A: It is the dimension of the model.
        self.d_model = d_model
        self.embedding = embedding
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

    def forward(self, sources, mask):
        sources = self.embedding(sources)

        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)

        return sources


class TransformerDecoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.embedding = embedding
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )
        # Q: What do embedding_dim and num_embedding mean here?
        # A: Embedding_dim is the dimension for each token's embedding while num_embeddings refers to the vocab size.
        self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        self.generator.weight = self.embedding.weight

    def forward(self, inputs, memory, memory_mask, inputs_mask=None, state=None):
        inputs = self.embedding(inputs)

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            # I: It is at this point that the embeddings from the last layer of the encoder are
            # sent into the decoder. memory indicates the embeddings from the last layer of the encoder.
            if state is None:
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask)
            else:
                layer_cache = state.layer_caches[layer_index]
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask, layer_cache)

                # I: I find it very interesting how the state is used here, I am not sure yet
                # but I think this is how the keys and values from the encoder and relayed to the
                # decoder.
                state.update_state(
                    layer_index=layer_index,
                    layer_mode="self-attention",
                    key_projected=decoder_layer.self_attention_layer.sublayer.key_projected,
                    value_projected=decoder_layer.self_attention_layer.sublayer.value_projected
                )

                state.update_state(
                    layer_index=layer_index,
                    layer_mode="memory-attention",
                    key_projected=decoder_layer.memory_attention_layer.sublayer.key_projected,
                    value_projected=decoder_layer.memory_attention_layer.sublayer.value_projected
                )

        # Q: I can't find any softmax function called on the linear layer here. Why?
        generated = self.generator(inputs)
        return generated, state

    def init_decoder_state(self, **args):
        return DecoderState()


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_prob, mode="self-attention"):
        super(MultiHeadAttention, self).__init__()

        # I: This is super cool, checking that the heads_counts can divide the model's dimension
        # wholly. Makes a lot of sense.
        assert d_model % heads_count == 0
        # Q: What is the difference between self-attention and memory attention?
        # A: Self attention here refers to attention calculated from a regular attention mechanism
        # using only embeddings generated using a text. Memory attention refers to attention that uses
        # external embeddings.
        assert mode in ("self-attention", "memory-attention")

        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode

        # Q: Why are creating attention head with only one type of weight?
        # A: Actually that is a very nice move, because somewhere in the code, this layer
        # is then broken into multiple heads. I find it now easier to work with in comparison to
        # creating multiple nn.Linear layers.
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.final_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        # Q: Why is softmax dim=3 being used here?
        # A: The dimension indicates the axis on which the softmax will be calculated. So the results of
        #  q * k when calculating attention are on dim=3
        self.softmax = nn.Softmax(dim=3)

        self.attention = None

        # Q: The code says these variables are for cache. But for caching what? When is the cache being used?
        # A: Used to store embeddings such as the key and value embeddings from the encoder
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        if layer_cache is None or layer_cache[self.mode] is None:
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:
            if self.mode == "self-attention":
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)

                # Q: What is the use case of concatenating cached key and value weights with generated model weights?
                key_projected = torch.cat([key_projected, layer_cache[self.mode]["key_projected"]], dim=1)
                value_projected = torch.cat([value_projected, layer_cache[self.mode]["value_projected"]], dim=1)
            elif self.mode == "memory-attention":
                # I: I find it interesting that the same key and value from the encoder are used on all layers of
                # multi-head attention (note, not masked multi-head) in the decoder.
                # I: I also find it interesting that it is the projection being used here i.e. the result of the
                # already done multiplication of the key weights and the vocab embeddings.
                # I: I have seen why now, it is useful during the prediction phase, since the keys and values from the
                # encoder are the same through all iterations, it is cached after the first projection where the cache
                # in that layer is None.
                key_projected = layer_cache[self.mode]["key_projected"]
                value_projected = layer_cache[self.mode]["value_projected"]

        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        # Q: Why are all three projections transposed?
        # A: The transposition done here is not to transpose the key matrix itself, instead it is to ensure
        # that all the rows are assigned to their proper heads.
        # For example:
        # tq
        # tensor([[[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
        #          [11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]],
        #         [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
        #          [11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]]])
        # tq.view(2, 2, 2, 5)
        # tensor([[[[ 1.,  2.,  3.,  4.,  5.],
        #           [ 6.,  7.,  8.,  9., 10.]],
        #          [[11., 12., 13., 14., 15.],
        #           [16., 17., 18., 19., 20.]]],
        #         [[[ 1.,  2.,  3.,  4.,  5.],
        #           [ 6.,  7.,  8.,  9., 10.]],
        #          [[11., 12., 13., 14., 15.],
        #           [16., 17., 18., 19., 20.]]]])
        # The above splits the matrix into multiple heads, but the rows still contain those multiple heads
        # Like [ 1.,  2.,  3.,  4.,  5.] should belong to head 1 and [ 6.,  7.,  8.,  9., 10.] should belong
        # to head 2, but they remain in the same section so far.
        # res.transpose(1, 2)
        # tensor([[[[ 1.,  2.,  3.,  4.,  5.],
        #           [11., 12., 13., 14., 15.]],
        #          [[ 6.,  7.,  8.,  9., 10.],
        #           [16., 17., 18., 19., 20.]]],
        # As you can see in the above now [ 1.,  2.,  3.,  4.,  5.] and [11., 12., 13., 14., 15.] belong to the same
        # head and hence are in the same section.
        # Q: Since the splitting of the heads is done after projection, does this not mean that the same
        # weights are used for all the heads?
        # A: Well, yes, but no. Since the linear is the length of the number of tokens before being split.
        # The weights from 0-100 for example will be different from weights from 100-200.
        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1, 2)

        attention_weights = self.scaled_dot_product(query_heads, key_heads)

        # Q: What type of value does mask expect as input?
        # I: I find it interesting that masking is applied before softmax is done, so that we don't end up
        # masking probability values by doing after softmax.
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)

        self.attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        # I: This is really nice, using transpose to return back to the initial shape.
        # Q: I find the use of contiguous here confusing
        # A: Methods like narrow, view, expand and transpose change the view of a tensor,
        # such that stride and offset operations look like the tensor actually changed to the shape described
        # in any of the operations metioned (or others like it), but the underlying tensor does not change.
        # Contiguous forces an underlying tensor to be created that exactly matches that shape.
        # A: What contiguous means: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        context_sequence = context_heads.transpose(1, 2).contiguous()
        # I: This is really nice too, restructuring back to the initial shape before dividing in multiple heads.
        context = context_sequence.view(batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.feed_forward(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask):
        sources = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)

        return sources


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerDecoderLayer, self).__init__()
        # Q: Interesting to see the memory-attention bit coming into play here
        # but it is not yet clear yet which is the masked multihead attention or the multihead attention
        # remember that the multihead attention takes into consideration the keys and values from the encoder.
        # A: Self attention is used for masked multi-head.
        self.self_attention_layer = Sublayer(MultiHeadAttention(
            heads_count, d_model, dropout_prob, mode="self-attention"), d_model)
        self.memory_attention_layer = Sublayer(MultiHeadAttention(
            heads_count, d_model, dropout_prob, mode="memory-attention"), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(
            d_ff, d_model, dropout_prob), d_model)

    def forward(self, inputs, memory, memory_mask, inputs_mask, layer_cache=None):
        inputs = self.self_attention_layer(inputs, inputs, inputs, inputs_mask, layer_cache)
        # I: I think it is confusing that memory is being passed in here if it is never used in the memory attention
        # layer. Instead it is the key and value projections existing in layer_cache that is being used.
        # Update: Actually, I did not understand properly the other time. The decoder has its own key and value weights,
        # What it needs are the embeddings from the last layer of the decoder and it multiplies that by its own key and value
        # weights.
        inputs = self.memory_attention_layer(inputs, memory, memory, memory_mask, layer_cache)
        inputs = self.pointwise_feedforward_layer(inputs)
        return inputs


class Sublayer(nn.Module):

    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()

        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class LayerNormalization(nn.Module):
    # A: Explanation of LayerNormalization
    # https://leimao.github.io/blog/Layer-Normalization/
    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNormalization, self).__init__()

        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class DecoderState:
    def __init__(self):
        self.previous_inputs = torch.tensor([])
        # I: I find it interesting how defaultdict is used here. In the past I have only just imply
        # passed in the data type like list, int, str into defaultdict. Exciting to see more explicit
        # data defaults provided.
        self.layer_caches = defaultdict(lambda: {"self-attention": None, "memory-attention": None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {
            "key_projected": key_projected,
            "value_projected": value_projected
        }

    # Q: What is beam update used for?
    def beam_update(self, positions):
        for layer_index in self.layer_caches:
            for mode in ("self-attention", "memory-attention"):
                if self.layer_caches[layer_index][mode] is not None:
                    for projection in self.layer_caches[layer_index][mode]:
                        cache = self.layer_caches[layer_index][mode][projection]
                        if cache is not None:
                            # In the post: https://discuss.pytorch.org/t/which-copy-is-better/56393/5
                            # The moderators say data.copy_ should not longer be used, so I am modifying
                            # cache.data.copy_(cache.data.index_select(0, positions))
                            # Q: Why is this copy here being done?
                            # A: It is replacing the cache tensor with the tensor resulting from the index_select
                            # positions here are the ids that have the most probability score for each token generation
                            # attempt during inference.
                            with torch.no_grad():
                                cache.copy_(cache.data.index_select(0, positions))