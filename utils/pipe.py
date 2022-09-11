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
