*This project has been created as part of the 42 curriculum by l-mulerf*

# Call Me Maybe: LLM Function Calling via Constrained Decoding

# Description
The "Call Me Maybe" project solves a fundamental problem in Generative AI: forcing a Small Language Model (SLM), specifically `Qwen3-0.6B` to output strictly structured, machine-executable JSON data. 

Instead of allowing the model to answer user prompts freely, this system identifies the correct function to call and extracts the required arguments based on a provided schema. By implementing a custom **Constrained Decoding Engine** and a **Contextual Scaffolding FSM (Finite State Machine)**, the system intercepts logits during text generation. This guarantees 100% adherence to the JSON schema without relying on the statistical probability of the model alone.

# Instructions

### Requirements
- Python 3.10+
- `uv` package manager

### Compilation & Installation
To install the project and its dependencies deterministically using `uv`, run:
```bash
make install
```

### Execution
To run the main pipeline using the default test files provided in the data/input directory:
```bash
make run
```

Optionally, you can specify custom input and output paths:
```bash
uv run python -m src --functions_definition <path_to_schema.json> --input <path_to_prompts.json> --output <path_to_results.json>
```

### Quality Assurance
To verify the code against flake8 and strict mypy type-checking rules:
```bash
make lint-strict
```

## The Bonus Part: Pure Python Tokenizer
The PDF outlines an optional bonus: "Recoding the tokenizer: avoiding direct use of `encode` and `decode` in the main code, instead using `get_logits_from_input_ids` and `get_path_to_vocabulary_json`".

### What is the Bonus?
By default, the provided `llm_sdk` uses the HuggingFace `AutoTokenizer` (written in Rust/C++) via the `model.encode()` method. For the bonus, we bypassed this entirely.

Inside `src/constrained_dec.py`, we implemented a *Greedy Longest Match Tokenizer* in pure Python. It reads the raw `vocab.json`, sorts all ~150,000 strings by length, and iteratively matches the longest possible substring from the user's prompt to generate Token IDs natively.

### How to test WITH the Bonus (Current State)

The repository is currently submitted with the Bonus enabled.
If you check `src/generator.py`, you will see lines like:
```bash
input_ids = self.mapper.encode(full_prompt)
```
This means we are successfully transforming text into numbers without touching the HuggingFace tokenizer.

### How to test WITHOUT the Bonus (Mandatory Only)

You can easily disable the custom tokenizer and revert to the SDK's default encoder.
Simply change the `encode` calls in `src/generator`.py from `self.mapper` to `self.model`:

```bash
# Change this (Bonus):
input_ids = self.mapper.encode(system_prompt)

# To this (Mandatory Only):
input_ids = self.model.encode(system_prompt).tolist()[0]

```



## Algorithm Explanation
The core approach utilizes a **Two-Phase Scaffolding Architecture**:
1.  **Phase 1 (Function Selection):** A Prefix Tree (Trie) is built in memory containing only valid function names from the schema. The LLM's generation is constrained to traverse only valid nodes in this Trie.
2. **Phase 2 (Argument Extraction / Scaffolding):** Instead of generating the entire JSON at once (which degrades SLM performance), we use Contextual Scaffolding. We build the JSON programmatically in Python `({"name": "fn_add", "parameters": {"a":`) and feed this exact string back to the LLM. The LLM only generates the specific value for the current key.
3. **Logit Masking:** During generation, we use Python's native `max()` function over a pre-cached subset of valid tokens (e.g., only numeric tokens for a `<number>` field) to select the highest-probability valid token in `O(V)` time.

## Design Decisions
1. **Strict Pydantic Compliance:** To rigidly follow the rule *"All classes must use pydantic for validation"*, literally every class in the project — including engines like `FunctionCaller` and `VocabularyMapper` — inherits from `pydantic.BaseModel`. `ConfigDict(arbitrary_types_allowed=True)` was used to safely pass the SDK's model instance through Pydantic's initialization validator without sacrificing runtime loop speed.
2. **O(1) Token Pre-Caching:** To avoid evaluating string rules on 150,000 tokens at every generation step, we pre-calculate which token IDs contain valid numbers or booleans during the `VocabularyMapper` initialization. This turns logit filtering into an instant `O(1)` subset lookup.
3. **The "Prompt Diet":** Feeding the entire schema into the LLM context for every single token caused the underlying model's attention mechanism (which operates at $O(N^2)$ complexity) to severely bottleneck the generation. We stripped the prompt to contain only the necessary instructions per phase, accelerating inference speeds dramatically.

## Performance Analysis
- **Accuracy:** The two-phase scaffolding paired with strict instructional prompts significantly improves semantic extraction for the 0.6B model.
- **Reliability: 100% valid JSON generation.** The constrained engine makes it mathematically impossible for the LLM to output malformed JSON syntax or hallucinate non-existent keys.
- Speed: By changing the logic from an $O(V log V)$ `sorted()` call to an $O(V)$ `max()` scan over cached sub-vocabularies, the pipeline processes all test prompts in ~2 to 3 minutes (well below the 5-minute requirement).

## Challenges Faced
- **BPE Tokenization Artifacts:** Subword tokenizers generate structural symbols (like `Ġ` for leading spaces and `Ċ` for newlines). This broke strict string validation initially. Solved by implementing an artifact cleaner in the `VocabularyMapper`.
- **Attention Amnesia:** The small 0.6B model lost context when generating long JSON structures, resulting in hallucinations (e.g., repeating the key name as the value). Solved via Contextual Scaffolding and targeted instructions ("*Extract the exact parameter values... DO NOT execute the function*").
- **Inference Bottleneck:** Generating tokens while keeping the entire function schema in the prompt caused an exponential slowdown due to the underlying neural network's mathematical complexity. Solved by feeding the model only the bare minimum context required for each specific phase (Contextual Scaffolding).

## Testing Strategy
The implementation was validated using standard prompts and aggressive edge-case testing (`data/input/edge_cases_test.json`), including:

- **Empty strings and large whitespace blocks** to test default fallback behavior.
- **Extreme numbers** like scientific notation (`1e-13`) and large negatives (`-999999.99`).
- **Invalid Input Files:** missing files or malformed JSONs are handled gracefully via `try/except` blocks in the orchestrator, ensuring the program never throws a raw stack trace.

## Example Usage

**Input: `"Replace all vowels in 'Programming is fun' with asterisks"`**

**Output generated in `function_calling_results.json`:**

```JSON
{
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
        "source_string": "Programming is fun",
        "regex": "aeiouAEIOU",
        "replacement": "****"
    }
}
```

# Resources

### Classic References & Documentation
- [RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format](https://datatracker.ietf.org/doc/html/rfc8259)
- [Byte-Pair Encoding Tokenization — Sennrich et al. (2015)](https://arxiv.org/abs/1508.07909)
- [Efficient Guided Generation for LLMs — Willard & Louf (2023)](https://arxiv.org/abs/2307.09702) — Theoretical foundation for FSM-based constrained decoding.
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [uv — Fast Python Package Manager](https://github.com/astral-sh/uv)

### AI Usage Declaration
In strict adherence to the project's AI Instructions, AI was **not** used to generate direct codebase solutions, nor was code blindly copy-pasted. Instead, AI was utilized as a **sparring partner and technical consultant** to brainstorm architectural patterns, understand the underlying mathematics of tokenization, and debug performance bottlenecks. I take full responsibility for every line of code in this repository and can explain its exact purpose and execution flow.

**AI Models Used:** Gemini 3.1 Pro Preview

**Specific Tasks & Prompt Examples:**

1. **Task: Understanding BPE Tokenizer Artifacts (Phase: Pure Python Tokenizer Bonus)**
   - *Prompt:* "I am building a custom tokenizer in pure Python using a `vocab.json` file. Explain how BPE tokenizers handle leading spaces and newlines, and why symbols like `Ġ` and `Ċ` appear in the vocabulary. How can I handle this cleanly before validating a JSON string?"
   - *Result:* Helped me design the `_clean_token_string` methodology inside the `VocabularyMapper` to sanitize tokens before FSM validation.

2. **Task: Performance Optimization (Phase: Generator Pipeline)**
   - *Prompt:* "My constrained decoding loop takes 15 minutes to run because I am calling `sorted()` on a list of 150,000 logits at every single step to find the best valid token. Since I cannot use external libraries like NumPy for the logic, how can I refactor this to an $O(V)$ complexity in pure Python?"
   - *Result:* Guided me toward pre-caching valid subset tokens (like `number_tokens`) and using Python's native `max()` with a lambda key, dropping the execution time to under 3 minutes.

3. **Task: Resolving LLM Hallucinations (Phase: Prompt Engineering)**
   - *Prompt:* "I am using a small 0.6B parameters model. Even though my FSM restricts it to output valid JSON, the model is suffering from 'attention amnesia' and outputting the key name as the value (e.g., `{'name': 'name'}`). How should I structure my system prompt to force it to extract the actual value from the user request?"
   - *Result:* Led to the discovery of the "Contextual Scaffolding" technique, where I programmatically build the JSON prefix in Python and instruct the model specifically to *extract* rather than *execute*.


## Important

The `Makefile` is optimized for the `42 School infrastructure` and automatically redirects caches and virtual environments to `/sgoinfre/$USER`. If you are not using a 42 campus machine, you may need to modify these paths according to your environment.