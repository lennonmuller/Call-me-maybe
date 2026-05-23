"""
Constrained Decoding Core.
Implements the Bonus Tokenizer and the Function Trie.
"""

import json
from typing import TypedDict
from llm_sdk import Small_LLM_Model  # type: ignore
from src.schema_models import FunctionDef


class TrieNode(TypedDict):
    children: dict[int, "TrieNode"]
    is_end: bool
    fn_name: str | None


class VocabularyMapper:
    """
    Handles vocabulary mapping and implements a Custom Tokenizer (Bonus Part)
    in pure Python without using model.encode() or model.decode().
    """
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.vocab_inverted = {v: k for k, v in self.vocab.items()}
        self.id_to_clean_token: dict[int, str] = {}

        for tid, raw_str in self.vocab_inverted.items():
            self.id_to_clean_token[tid] = raw_str.replace('Ġ', ' ').replace('Ċ', '\n').replace('ċ', '\n')

        self._sorted_tokens = sorted(self.vocab.items(), key=lambda x: len(x[0]), reverse=True)

        self.number_tokens: list[int] = []
        self.boolean_tokens: list[int] = []

        for tid, t_str in self.id_to_clean_token.items():
            if all(c in "0123456789.-+eE \t" for c in t_str) or any(c in ",\n}" for c in t_str):
                self.number_tokens.append(tid)
            s = t_str.strip()
            if s in [
                    "", "t", "tr", "tru", "true",
                    "f", "fa", "fal", "fals", "false",
                    ] or any(c in ",\n}" for c in t_str):
                self.boolean_tokens.append(tid)

    def token_to_str(self, token_id: int) -> str:
        return self.id_to_clean_token.get(token_id, "")

    def encode(self, text: str) -> list[int]:
        """BONUS: Pure Python Tokenizer using Greedy Longest Match."""
        ids: list[int] = []
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

    def decode(self, ids: list[int]) -> str:
        """BONUS: Pure Python Decoder."""
        return "".join([self.token_to_str(tid) for tid in ids])


class FunctionTrie:
    def __init__(self) -> None:
        self.root: TrieNode = {"children": {}, "is_end": False, "fn_name": None}

    def insert(self, tokens: list[int], fn_name: str) -> None:
        current = self.root
        for token in tokens:
            if token not in current["children"]:
                current["children"][token] = {"children": {}, "is_end": False, "fn_name": None}
            current = current["children"][token]
        current["is_end"] = True
        current["fn_name"] = fn_name

    def get_valid_tokens(self, tokens_generated: list[int]) -> list[int]:
        current = self.root
        for token in tokens_generated:
            if token in current["children"]:
                current = current["children"][token]
            else:
                return []
        return list(current["children"].keys())

    def is_function_complete(self, tokens: list[int]) -> bool:
        current = self.root
        for token in tokens:
            if token not in current["children"]:
                return False
            current = current["children"][token]
        return current["is_end"]

    def get_fn_name(self, tokens: list[int]) -> str | None:
        current = self.root
        for token in tokens:
            if token not in current["children"]:
                return None
            current = current["children"][token]
        return current["fn_name"]


def build_trie(functions: list[FunctionDef], mapper: VocabularyMapper) -> FunctionTrie:
    """Builds the FSM Trie using the custom tokenizer."""
    trie = FunctionTrie()
    for function in functions:
        tokens = mapper.encode(function.name)
        trie.insert(tokens, function.name)
    return trie
