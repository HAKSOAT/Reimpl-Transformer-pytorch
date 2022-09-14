import torch

from utils.constants import SPECIAL_TOKENS_INDEX, PAD_TOKEN


def shared_tokens_generator(dataset):
    for source, target in dataset:
        for token in source:
            yield token

        for token in target:
            yield token


def source_tokens_generator(dataset):
    for source, target in dataset:
        for token in source:
            yield token


def target_tokens_generator(dataset):
    for source, target in dataset:
        for token in target:
            yield token


def input_target_collate_fn(batch):
    sources_lengths = []
    inputs_lengths = []
    targets_lengths = []

    for sources, inputs, targets in batch:
        sources_lengths.append(len(sources))
        inputs_lengths.append(len(inputs))
        targets_lengths.append(len(targets))

    sources_max_length = max(sources_lengths)
    inputs_max_length = max(inputs_lengths)
    targets_max_length = max(targets_lengths)

    sources_padded = []
    inputs_padded = []
    targets_padded = []

    for sources, inputs, targets in batch:
        sources_padded.append(
            sources + [SPECIAL_TOKENS_INDEX[PAD_TOKEN]] * (sources_max_length - len(sources))
        )
        inputs_padded.append(
            inputs + [SPECIAL_TOKENS_INDEX[PAD_TOKEN]] * (inputs_max_length - len(inputs))
        )
        targets_padded.append(
            targets + [SPECIAL_TOKENS_INDEX[PAD_TOKEN]] * (targets_max_length - len(targets))
        )

    sources_tensor = torch.tensor(sources_padded)
    inputs_tensor = torch.tensor(inputs_padded)
    targets_tensor = torch.tensor(targets_padded)

    return sources_tensor, inputs_tensor, targets_tensor
