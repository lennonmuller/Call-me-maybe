"""
Main entry point for the Function Calling project.
Integrates the LLM SDK with the Constrained Decoding engine to process prompts
and generate structured JSON outputs.
"""

import argparse
import sys
import json
from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.schema_models import FunctionDef, ParameterDef
from src.vocabulary import Vocabulary
from src.constraint_engine import JSONConstraintEngine


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments according to project requirements."""
    parser = argparse.ArgumentParser(
        description="LLM Function Calling com Constrained Decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing test prompts."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the output JSON will be saved."
    )
    return parser.parse_args()


def load_functions(filepath: str) -> list[FunctionDef]:
    """Loads and validates functio definitions from a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        functions = []
        for item in data:
            # converting raw dicts to pydantic paramerdef manually
            params = {
                k: ParameterDef(type=v['type'])
                for k, v in item.get('parameters', {}).items()
            }
            functions.append(FunctionDef(
                name=item['name'],
                description=item['description'],
                parameters=params
            ))
        return functions
    except Exception as e:
        raise ValueError(f"Failed to load function definitions: {e}")


def load_prompts(filepath: str) -> list[dict[str, str]]:
    """Loads natural language prompts from the input JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load prompts: {e}")
    

def generate_constrained_json(
        prompt: str,
        model: Small_LLM_Model,
        engine: JSONConstraintEngine,
        max_tokens: int = 300
) -> str:
    """
    Core function for Contrained Decoding. Generates text token by token,
    filtering logits throung the Constraint Engine to guarantee valid JSON.
    """
    funcs_dict = []
    for f in engine.functions:
        params = {k: {"type": v.type} for k, v in f.parameters.items()}
        funcs_dict.append({"name": f.name, "description": f.description, "parameters": params})
    
    funcs_str = json.dumps(funcs_dict, indent=2)

    system_prompt = (
        "You are an expert AI function calling assistant. "
        "You must select the correct function from the list below based on the user's request. "
        "Output ONLY a valid JSON object matching the chosen function's exact schema. Do not add any conversational text.\n\n"
        f"AVAILABLE FUNCTIONS:\n{funcs_str}"
    )
    full_prompt = f"{system_prompt}\nUser request: {prompt}\nJSON output:\n"

    # tokenization
    input_tensor = model.encode(full_prompt)
    current_ids = input_tensor[0].tolist()

    generated_text = ""

    # generation loop
    for step in range(max_tokens):
        next_token_logits = model.get_logits_from_input_ids(current_ids)

        scored_tokens = sorted(
            [(score, token_id) for token_id, score in enumerate(next_token_logits)],
            reverse=True,
            key=lambda x: x[0]
        )

        best_valid_token_id = -1

        # token selection
        for _, token_id in scored_tokens:
            if engine.is_token_valid(generated_text, token_id):
                best_valid_token_id = token_id
                break

        if best_valid_token_id == -1:
            print(f"Warning: No valid tokens found at step {step}. Halting.", file=sys.stderr)
            break

        # update state
        current_ids.append(best_valid_token_id)

        raw_token_str = engine.vocabulary.get_token_string(best_valid_token_id)
        cleaned_str = engine._clean_token_string(raw_token_str)
        generated_text += cleaned_str

        # early exit
        try:
            json.loads(generated_text)
            break # json completo
        except json.JSONDecodeError:
            pass # continua o loop

    return generated_text


def main() -> None:
    """Main execution"""
    args = parse_arguments()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing components...")
    try:
        functions = load_functions(args.functions_definition)
        prompts_data = load_prompts(args.input)

        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()

        vocabulary = Vocabulary(vocab_path)
        constraint_engine = JSONConstraintEngine(functions, vocabulary)

    except Exception as e:
        print(f"Initialization Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_prompts = len(prompts_data)

    print(f"Starting processing of {total_prompts} prompts...")

    for i, item in enumerate(prompts_data, 1):
        prompt_text = item.get("prompt", "")
        print(f"[{i}/{total_prompts}] Processing: '{prompt_text}'")

        try:
            raw_json_str = generate_constrained_json(prompt_text, model, constraint_engine)
            parsed_json = json.loads(raw_json_str)

            result_obj = {
                "prompt": prompt_text,
                "name": parsed_json.get("name", ""),
                "parameters": parsed_json.get("parameters", {})
            }
            results.append(result_obj)
        except Exception as e:
            print(f"Error processing prompt '{prompt_text}': {e}", file=sys.stderr)
            continue
    
    print(f"Saving results to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print("Done. Sucess!")
    except IOError as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro inesperado: {e}", file=sys.stderr)
        sys.exit(1)
