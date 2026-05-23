"""
Generation pipeline using Contextual Scaffolding for ultra-fast generation.
"""

from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from src.constrained_dec import VocabularyMapper, FunctionTrie
from src.schema_models import FunctionDef


class FunctionCaller:
    def __init__(self,
                 model: Small_LLM_Model,
                 mapper: VocabularyMapper,
                 trie: FunctionTrie,
                 functions: list[FunctionDef]) -> None:
        self.model = model
        self.mapper = mapper
        self.trie = trie
        self.functions = functions

    def select_function(self, prompt: str) -> str | None:
        funcs_info = "\n".join([f"- {f.name}: {f.description}" for f in self.functions])
        system_prompt = f"System: Select the correct function.\nFunctions:\n{funcs_info}" \
                        f"\nUser: {prompt}\nJSON:\n{{\n  \"name\": \""

        input_ids = self.mapper.encode(system_prompt)
        tokens_generated: list[int] = []

        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)
            valid_tokens = self.trie.get_valid_tokens(tokens_generated)

            if not valid_tokens:
                return None

            best_token = max(valid_tokens, key=lambda t: logits[t])
            tokens_generated.append(best_token)
            input_ids.append(best_token)

            if self.trie.is_function_complete(tokens_generated):
                break

        return self.trie.get_fn_name(tokens_generated)

    def _generate_number(self, full_prompt: str) -> str:
        input_ids = self.mapper.encode(full_prompt)
        val_str = ""
        while len(val_str) < 15:
            logits = self.model.get_logits_from_input_ids(input_ids)

            best_token = max(self.mapper.number_tokens, key=lambda t: logits[t])
            t_str = self.mapper.token_to_str(best_token)

            if any(c in ",\n}" for c in t_str):
                for char in t_str:
                    if char in ",\n}":
                        break
                    val_str += char
                break

            input_ids.append(best_token)
            val_str += t_str
        return val_str.strip()

    def _generate_string(self, full_prompt: str) -> str:
        input_ids = self.mapper.encode(full_prompt)
        val_str = ""
        vocab_size = len(self.mapper.vocab)

        while len(val_str) < 100:
            logits = self.model.get_logits_from_input_ids(input_ids)

            best_token = max(range(vocab_size), key=lambda t: logits[t])
            t_str = self.mapper.token_to_str(best_token)

            if '"' in t_str:
                val_str += t_str.split('"')[0]
                break

            input_ids.append(best_token)
            val_str += t_str
        return val_str

    def _generate_boolean(self, full_prompt: str) -> str:
        input_ids = self.mapper.encode(full_prompt)
        val_str = ""
        while len(val_str) < 6:
            logits = self.model.get_logits_from_input_ids(input_ids)

            best_token = max(self.mapper.boolean_tokens, key=lambda t: logits[t])
            t_str = self.mapper.token_to_str(best_token)

            if any(c in ",\n}" for c in t_str):
                break

            input_ids.append(best_token)
            val_str += t_str
            if val_str.strip() in ["true", "false"]:
                break
        return val_str.strip()

    def process_prompt(self, prompt: str) -> dict[str, Any]:
        fn_name = self.select_function(prompt)
        if not fn_name:
            raise ValueError(f"Could not select function for: {prompt}")

        selected_function = next((f for f in self.functions if f.name == fn_name), None)
        if not selected_function:
            raise ValueError(f"Function {fn_name} not found.")

        args: dict[str, Any] = {}
        json_context = f'{{\n  "name": "{fn_name}",\n  "parameters": {{\n'

        keys = list(selected_function.parameters.keys())
        for i, key in enumerate(keys):
            param_def = selected_function.parameters[key]
            json_context += f'    "{key}": '

            full_prompt = f"Task: Extract the exact value for '{key}' from the user request.\n" \
                          f"User: {prompt}\nJSON:\n{json_context}"

            if param_def.type == "string":
                json_context += '"'
                full_prompt += '"'
                val_str = self._generate_string(full_prompt)
                args[key] = val_str
                json_context += f'{val_str}"'

            elif param_def.type == "number":
                val_str = self._generate_number(full_prompt)
                try:
                    args[key] = float(val_str) if '.' in val_str else int(val_str)
                except ValueError:
                    args[key] = 0
                json_context += val_str

            elif param_def.type == "boolean":
                val_str = self._generate_boolean(full_prompt)
                args[key] = (val_str == "true")
                json_context += val_str

            if i < len(keys) - 1:
                json_context += ',\n'
            else:
                json_context += '\n  }\n}'

        return {
            "prompt": prompt,
            "name": fn_name,
            "parameters": args
        }
