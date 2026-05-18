"""
Constraint Engine for Constrained Decoding.
It uses 'Deterministic Rails' (Templates) to enforce an exact structure and schema,
reducing computational complexity to O(N).
"""

from src.schema_models import FunctionDef
from src.vocabulary import Vocabulary


class JSONConstraintEngine:
    """
    Controls the state of JSON generation and decides
    if an specific token can or not be added to the actual sequence
    """

    def __init__(self, functions: list[FunctionDef], vocabulary: Vocabulary):
        self.functions = functions
        self.vocabulary = vocabulary

        self.templates = self._build_all_templates()

    def _build_all_templates(self) -> list[list[str]]:
        """
        Pre calculates the template allowed for each function.
        This transforms the JSON into a predictable sequence of text blocks.
        """
        templates = []
        for func in self.functions:
            parts = []
            parts.append(f'{{"name": "{func.name}", "parameters": {{')

            keys = sorted(func.parameters.keys())
            for i, key in enumerate(keys):
                param_type = func.parameters[key].type
                if i > 0:
                    parts.append(', ')
                parts.append(f'"{key}": ')

                parts.append(f'<{param_type}>')

            parts.append('}}')
            templates.append(parts)
        return templates

    def _clean_token_string(self, token_str: str) -> str:
        """
        Clean special strings generated for tokenizers (BPE/SentencePiece)
        The character 'Ġ' (G with a dot) is often used to represent a space
        preceding a word. 'Ċ' or 'ċ' represent line breaks."""
        cleaned = token_str.replace('Ġ', ' ').replace('Ċ', '\n').replace('ċ', '\n')
        return cleaned

    def is_token_valid(self, current_text: str, token_id: int) -> bool:
        """
        Evaluates whether adding the token keeps the text as a valid prefix."""
        token_str = self.vocabulary.get_token_string(token_id)

        if not token_str:
            return False

        cleaned_token = self._clean_token_string(token_str)
        candidate_text = current_text + cleaned_token
        for template in self.templates:
            if self._matches_template_prefix(candidate_text, template):
                return True
        return False

    def _matches_template_prefix(self, text: str, template: list[str]) -> bool:
        """
        check is the text generated so far strictly follows a template.
        """
        text_idx = 0

        for part in template:
            if text_idx >= len(text):
                return True

            if part == "<string>":
                text_idx = self._consume_string(text, text_idx)
                if text_idx == -1:
                    return False

            elif part == "<number>":
                text_idx = self._consume_number(text, text_idx)
                if text_idx == -1:
                    return False

            elif part == "<boolean>":
                text_idx = self._consume_boolean(text, text_idx)
                if text_idx == -1:
                    return False

            else:
                match_len = min(len(part), len(text) - text_idx)
                if text[text_idx: text_idx + match_len] != part[:match_len]:
                    return False
                text_idx += match_len

        # se consumiu todo o template e sobrou texto: invalido
        return text_idx >= len(text)

    def _consume_string(self, text: str, idx: int) -> int:
        """
        Consumes a JSON string value, return a new index, or -1 if invalid.
        """
        # if the string start and is not aspas is invalid
        if text[idx] != '"':
            return -1

        # Search for the end of string, ignoring "extra aspas"
        i = idx + 1
        escape = False
        while i < len(text):
            if escape:
                escape = False
            elif text[i] == '\\':
                escape = True
            elif text[i] == '"':
                return i + 1
            i += 1

        return len(text)

    def _consume_number(self, text: str, idx: int) -> int:
        """
        It consumes a JSON number (allows digits, decimal point, and minus sign).
        """
        allowed_chars = set("0123456789.-")
        i = idx
        while i < len(text):
            if text[i] in ",}":
                return i
            if text[i] not in allowed_chars:
                return -1
            i += 1
        return len(text)

    def _consume_boolean(self, text: str, idx: int) -> int:
        """
        It consumes a boolean value(true or false)
        """
        remaining = text[idx:]
        if "true".startswith(remaining) or "false".startswith(remaining):
            return len(text)

        # "jump" the character if starts with
        if remaining.startswith("true"):
            return idx + 4
        if remaining.startswith("false"):
            return idx + 5

        return -1
