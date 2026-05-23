"""
This module is responsible for loading and managing the LLM vocabulary.
Implements pure Python greedy tokenization for the bonus requirement.
"""

import json
from typing import Optional


class Vocabulary:
    """
    Loads the model vocabulary JSON file and creates efficient mappings
    for use during Constrained Decoding.
    """

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.id_to_token: dict[int, str] = {}
        self.str_to_id: dict[str, int] = {}
        self.id_to_clean_token: dict[int, str] = {}
        self._sorted_tokens: list[tuple[str, int]] = []
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """
        Read the vocabulary file and fill the reverse dictionary.
        Usually, vocabulary files are {"token": id}.
        """

        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                token_to_id: dict[str, int] = json.load(f)

            self.id_to_token = {v: k for k, v in token_to_id.items()}
            self.str_to_id = token_to_id

            self._sorted_tokens = sorted(self.str_to_id.items(), key=lambda x: len(x[0]), reverse=True)

            # clean all the tokens in the initialization (O(V) exec 1 time)
            for t_id, t_str in self.id_to_token.items():
                self.id_to_clean_token[t_id] = t_str.replace('Ġ', ' ').replace('Ċ', '\n').replace('ċ', '\n')
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
        except json.JSONDecodeError:
            raise ValueError(f"The file {self.vocab_path} is not a valid JSON.")

    def encode(self, text: str) -> list[int]:
        """Pure Python Greedy Longest Match Tokenizer"""
        ids = []
        encoded_text = text.replace(' ', 'Ġ').replace('\n', 'Ċ')

        while encoded_text:
            matched = False
            for token_str, token_id in self._sorted_tokens:
                if encoded_text.startswith(token_str):
                    ids.append(token_id)
                    encoded_text = encoded_text[len(token_str):]
                    matched = True
                    break
                if not matched:
                    encoded_text = encoded_text[1:]
        return ids

    def get_token_string(self, token_id: int) -> str:
        """Returns the raw literal string associated with a token ID."""
        return self.id_to_token.get(token_id, "")

    def get_clean_token_string(self, token_id: int) -> Optional[str]:
        """Returns the pre-cleaned token string O(1)"""
        return self.id_to_clean_token.get(token_id)

    def get_vocab_size(self) -> int:
        """
        Returns the total number of tokens in the vocabulary.
        """
        return len(self.id_to_token)
