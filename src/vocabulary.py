"""
This module is responsible for loading and managing the LLM vocabulary.
This allows us to translate the generated IDs into strings for structural validation.
"""

import json


class Vocabulary:
    """
    Loads the model vocabulary JSON file and creates efficient mappings
    for use during Constrained Decoding.
    """

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.id_to_token: dict[int, str] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """
        Read the vocabulary file and fill the reverse dictionary.
        Usually, vocabulary files are {"token": id}.
        We need to reverse that
        """

        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                token_to_id: dict[str, int] = json.load(f)

            self.id_to_token = {v: k for k, v in token_to_id.items()}
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
        except json.JSONDecodeError:
            raise ValueError(f"The file {self.vocab_path} is not a valid JSON.")

    def get_token_string(self, token_id: int) -> str:
        """
        Returns the literal string associated to a ID token
        """
        if token_id not in self.id_to_token:
            return ""

        token_str = self.id_to_token[token_id]
        return token_str

    def get_vocab_size(self) -> int:
        """
        Returns the total number of tokens on the vocabulary.
        """
        return len(self.id_to_token)
