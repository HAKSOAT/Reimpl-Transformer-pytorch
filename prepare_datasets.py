from argparse import ArgumentParser

from datasets import (
    TranslationDataset, TranslationDatasetOnTheFly, TokenizedTranslationDataset,
    IndexedInputTargetTranslationDataset, IndexedInputTargetTranslationDatasetOnTheFly
)
from dictionaries import IndexDictionary
from utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator

parser = ArgumentParser("Prepare datasets")
parser.add_argument("--train_source", type=str, default="data/example/raw/src-train.txt")
parser.add_argument("--train_target", type=str, default="data/example/raw/tgt-train.txt")
parser.add_argument("--val_source", type=str, default="data/example/raw/src-val.txt")
parser.add_argument("--val_target", type=str, default="data/example/raw/tgt-val.txt")
# Q: What does save data dir do?
# A: It decided where the combined source and target language dataset is saved.
parser.add_argument("--save_data_dir", type=str, default="data/example/processed")
# Q: What does share dictionary do?
# A: It allows the source and target to be in the same file.
parser.add_argument("--share_dictionary", type=bool, default=False)


args = parser.parse_args()

TranslationDataset.prepare(args.train_source, args.train_target, args.val_source, args.val_target, args.save_data_dir)
# Q: How is TranslationDataset different from TranslationDatasetOnTheFly?
# A: OnTheFly works on datasets that have train and validation for both source and target files (when source and target
# are different). While TranslationDataset works on train and validation when source and target texts are both in
# one file. Also on the fly has a set file location, so you don't have to pass in a file location.
translation_dataset = TranslationDataset(args.save_data_dir, "train")
translation_dataset_on_the_fly = TranslationDatasetOnTheFly("train")
assert translation_dataset[0] == translation_dataset_on_the_fly[0]

tokenized_dataset = TokenizedTranslationDataset(args.save_data_dir, "train")

if args.share_dictionary:
    source_generator = shared_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode="source")
    target_generator = shared_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode="target")
else:
    source_generator = source_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode="source")
    target_generator = target_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode="target")

source_dictionary.save(args.save_data_dir)
target_dictionary.save(args.save_data_dir)

IndexedInputTargetTranslationDataset.prepare(args.save_data_dir, source_dictionary, target_dictionary)
indexed_translation_dataset = IndexedInputTargetTranslationDataset(args.save_data_dir, "train")
indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly(
    "train", source_dictionary, target_dictionary)
assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

print("Done datasets preparation.")
