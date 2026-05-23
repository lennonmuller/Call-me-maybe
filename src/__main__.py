"""
Main entry point for the Function Calling project.
Integrates the LLM SDK with the Constrained Decoding engine to process prompts.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import cast

from llm_sdk import Small_LLM_Model  # type: ignore
from src.schema_models import FunctionDef, ParameterDef
from src.constrained_dec import VocabularyMapper, build_trie
from src.generator import FunctionCaller


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    return parser.parse_args()


def load_functions(filepath: str) -> list[FunctionDef]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [FunctionDef(name=i['name'], description=i['description'],
            parameters={k: ParameterDef(type=v['type']) for k, v in i.get('parameters', {}).items()}) for i in data]


def load_prompts(filepath: str) -> list[dict[str, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return cast(list[dict[str, str]], json.load(f))


def main() -> None:
    args = parse_arguments()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Initializing components...")
    try:
        functions = load_functions(args.functions_definition)
        prompts_data = load_prompts(args.input)

        model = Small_LLM_Model()
        mapper = VocabularyMapper(model)  # mapper recebe model
        trie = build_trie(functions, mapper)  # trie recebe mapper
        caller = FunctionCaller(model, mapper, trie, functions)
    except Exception as e:
        print(f"Initialization Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_prompts = len(prompts_data)
    print(f"Processing {total_prompts} prompts...")

    start_time = time.time()
    for i, item in enumerate(prompts_data, 1):
        prompt_text = item.get("prompt", "")
        print(f"[{i}/{total_prompts}] Processing: '{prompt_text}'")
        try:
            res = caller.process_prompt(prompt_text)
            results.append(res)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f} seconds.")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
