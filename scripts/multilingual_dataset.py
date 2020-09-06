import hashlib
import logging
import os
import pickle
import time
from typing import List, Tuple

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class MultilingualTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            dataset_files: List[Tuple[int, str]],
            block_size: int,
            use_token_type_ids=True,
            overwrite_cache=False,
    ):

        self.examples = []
        for language_id, file_path in dataset_files:
            assert os.path.isfile(file_path)

            block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

            directory, filename = os.path.split(file_path)

            m = hashlib.md5()
            m.update(tokenizer.cache_id.encode())
            m.update(str(block_size).encode())
            m.update(file_path.encode())
            h = m.hexdigest()

            cached_features_file = os.path.join(
                directory, 'cached_{}'.format(h),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):
                language_examples = []

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        language_examples = pickle.load(handle)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )

                else:
                    logger.info(f"Creating features from dataset file at {directory}")

                    with open(file_path, encoding="utf-8") as f:
                        text = f.read()

                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                        tokens = tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                        # self.examples.append(
                        #     [tokens, [language_id]*len(tokens)]
                        # )
                        language_examples.append(tokens)
                    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should loook for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    start = time.time()
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(language_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
                if use_token_type_ids:
                    # add language IDs
                    self.examples += [[tokens, [language_id]*len(tokens)] for tokens in language_examples]
                else:
                    self.examples += tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
