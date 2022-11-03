import os

from utils.constants import INDEX_FILE_SEP, FILE_SEP, START_TOKEN, END_TOKEN, SPECIAL_TOKENS_INDEX

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_TYPES = ("train", "val")


# Q: There are a lot of Object and ObjectOnTheFly classes in this file. Why not just make the on the fly
# classes inherit from the ones not on the fly?
# A: I really feel that is the best way to approach this as it makes it easier to think about the ton of classes
# available in this file. But it will need quite some thinking to pull through as many of the instance methods
# are dependent on varying instance attributes initialized in the __init__ method.

class TranslationDataset:
    # Q: What is the purpose of the limit?
    # A: My understanding is that it limits the number of documents (lines) to be read from the dataset.
    def __init__(self, data_dir, phase, limit=None):
        # TODO: Confirm that this assert works.
        assert phase in DATASET_TYPES, f"Dataset phase must be one of {DATASET_TYPES}"

        self.limit = limit

        self.data = []
        with open(os.path.join(data_dir, f"raw-{phase}.txt"), encoding="utf-8") as file:
            for line in file:
                self.data.append(tuple(line.strip().split(FILE_SEP)))

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            return IndexError()

        return self.data[item]

    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else:
            return self.limit

    @staticmethod
    def prepare(train_source, train_target, val_source, val_target, save_data_dir):
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)

        for phase in ("train", "val"):
            if phase == "train":
                source_filepath = train_source
                target_filepath = train_target
            else:
                source_filepath = val_source
                target_filepath = val_target

            with open(source_filepath, encoding="utf-8") as source_file:
                source_data = source_file.readlines()

            with open(target_filepath, encoding="utf-8") as target_filepath:
                target_data = target_filepath.readlines()

            save_path = os.path.join(save_data_dir, f"raw-{phase}.txt")
            with open(save_path, "w", encoding="utf-8") as file:
                for source_line, target_line in zip(source_data, target_data):
                    source_line = source_line.strip()
                    target_line = target_line.strip()
                    line = f"{source_line}{FILE_SEP}{target_line}\n"
                    file.write(line)


class TranslationDatasetOnTheFly:
    def __init__(self, phase, limit=None):
        assert phase in DATASET_TYPES, f"Dataset phase must be either {DATASET_TYPES}"

        self.limit = limit

        if phase == "train":
            source_filepath = os.path.join(BASE_DIR, "data", "example", "raw", "src-train.txt")
            target_filepath = os.path.join(BASE_DIR, "data", "example", "raw", "tgt-train.txt")
        elif phase == "val":
            source_filepath = os.path.join(BASE_DIR, "data", "example", "raw", "src-val.txt")
            target_filepath = os.path.join(BASE_DIR, "data", "example", "raw", "tgt-val.txt")
        else:
            raise NotImplementedError()

        with open(source_filepath, encoding="utf-8") as source_file:
            self.source_data = source_file.readlines()

        with open(target_filepath, encoding="utf-8") as target_file:
            self.target_data = target_file.readlines()

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()

        source = self.source_data[item].strip()
        target = self.target_data[item].strip()
        return source, target

    def __len__(self):
        if self.limit is None:
            return len(self.source_data)
        else:
            return self.limit


class TokenizedTranslationDataset(TranslationDataset):
    def __getitem__(self, item):
        raw_source, raw_target = super(TokenizedTranslationDataset, self).__getitem__(item)
        tokenized_source = raw_source.split()
        tokenized_target = raw_target.split()
        return tokenized_source, tokenized_target


class InputTargetTranslationDataset:
    def __init__(self, data_dir, phase, limit=None):
        self.tokenized_dataset = TokenizedTranslationDataset(data_dir, phase, limit)

    def __getitem__(self, item):
        tokenized_source, tokenized_target = self.tokenized_dataset[item]
        full_target = [START_TOKEN] + tokenized_target + [END_TOKEN]
        inputs = full_target[:-1]
        targets = full_target[1:]
        return tokenized_source, inputs, targets

    def __len__(self):
        return len(self.tokenized_dataset)


class IndexedInputTargetTranslationDataset:
    def __init__(self, data_dir, phase, vocabulary_size=None, limit=None):
        self.data = []
        unknownify = lambda index: index if index < vocabulary_size else SPECIAL_TOKENS_INDEX["UNK_TOKEN"]
        with open(os.path.join(data_dir, f"indexed-{phase}.txt")) as file:
            for line in file:
                sources, inputs, targets = line.strip().split(FILE_SEP)
                if vocabulary_size is not None:
                    indexed_sources = [unknownify(int(index)) for index in sources.strip().split(INDEX_FILE_SEP)]
                    indexed_inputs = [unknownify(int(index)) for index in inputs.strip().split(INDEX_FILE_SEP)]
                    indexed_targets = [unknownify(int(index)) for index in targets.strip().split(INDEX_FILE_SEP)]
                else:
                    indexed_sources = [int(index) for index in sources.strip().split(INDEX_FILE_SEP)]
                    indexed_inputs = [int(index) for index in inputs.strip().split(INDEX_FILE_SEP)]
                    indexed_targets = [int(index) for index in targets.strip().split(INDEX_FILE_SEP)]

                self.data.append((indexed_sources, indexed_inputs, indexed_targets))
                if limit is not None and len(self.data) >= limit:
                    break

        self.vocabulary_size = vocabulary_size
        self.limit = limit

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()

        indexed_sources, indexed_inputs, indexed_targets = self.data[item]
        return indexed_sources, indexed_inputs, indexed_targets

    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else:
            return self.limit

    @staticmethod
    def preprocess(source_dictionary):

        def preprocess_function(source):
            source_tokens = source.strip().split()
            indexed_source = source_dictionary.index_sentence(source_tokens)
            return indexed_source

        return preprocess_function

    @staticmethod
    def prepare(data_dir, source_dictionary, target_dictionary):

        join_indexes = lambda indexes: INDEX_FILE_SEP.join(str(index) for index in indexes)
        for phase in DATASET_TYPES:
            input_target_dataset = InputTargetTranslationDataset(data_dir, phase)

            with open(os.path.join(data_dir, f"indexed-{phase}.txt"), "w") as file:
                for sources, inputs, targets in input_target_dataset:
                    indexed_sources = join_indexes(source_dictionary.index_sentence(sources))
                    # Q: What benefit does inputs here serve?
                    # Given the context that it only uses the target containing the start token but missing the
                    # end token, but targets has the target missing the start token but containing the end token.
                    # A: My assumption is that it will be useful for the decoder part of the transformer architecture
                    # where the input is the value at time `t` and the decoder is expected to predict the value at
                    # time `t+1`.
                    indexed_inputs = join_indexes(target_dictionary.index_sentence(inputs))
                    indexed_targets = join_indexes(target_dictionary.index_sentence(targets))
                    file.write(f"{indexed_sources}{FILE_SEP}{indexed_inputs}{FILE_SEP}{indexed_targets}\n")


class TokenizedTranslationDatasetOnTheFly:
    def __init__(self, phase, limit=None):
        self.raw_dataset = TranslationDatasetOnTheFly(phase, limit)

    def __getitem__(self, item):
        raw_source, raw_target = self.raw_dataset[item]
        tokenized_source = raw_source.split()
        tokenized_target = raw_target.split()
        return tokenized_source, tokenized_target

    def __len__(self):
        return len(self.raw_dataset)


class InputTargetTranslationDatasetOnTheFly:
    def __init__(self, phase, limit):
        self.tokenized_dataset = TokenizedTranslationDatasetOnTheFly(phase, limit)

    def __getitem__(self, item):
        tokenized_source, tokenized_target = self.tokenized_dataset[item]
        full_target = [START_TOKEN] + tokenized_target + [END_TOKEN]
        inputs = full_target[:-1]
        targets = full_target[1:]
        return tokenized_source, inputs, targets

    def __len__(self):
        return len(self.tokenized_dataset)


class IndexedInputTargetTranslationDatasetOnTheFly:
    def __init__(self, phase, source_dictionary, target_dictionary, limit=None):
        self.input_target_dataset = InputTargetTranslationDatasetOnTheFly(phase, limit)
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary

    def __getitem__(self, item):
        source, inputs, targets = self.input_target_dataset[item]
        indexed_source = self.source_dictionary.index_sentence(source)
        indexed_inputs = self.target_dictionary.index_sentence(inputs)
        indexed_targets = self.target_dictionary.index_sentence(targets)

        return indexed_source, indexed_inputs, indexed_targets

    def __len__(self):
        return len(self.input_target_dataset)
