import torch
from dataclasses import dataclass
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


@dataclass
class MultilingualDataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = self._tensorize_batch([example[0] for example in examples])
        token_type_ids = self._tensorize_batch([example[1] for example in examples])
        labels = input_ids.clone().detach()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels, "token_type_ids": token_type_ids}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
