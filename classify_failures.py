import argparse
import os
import sys
import time
from typing import Optional

import pandas as pd
from openai import OpenAI


FAILURE_LABELS = [
    "Environment Disturbance / Unable to Detect Change",
    "Instruction Error",
    "Catastrophic Forgetting",
    "False Assumptions",
    "Planning Errors (Sub-plan & Action)",
    "History Error Accumulation",
    "Memory Limitations",
]


SYSTEM_PROMPT = """You are an expert judge for failure analysis of LLM agents on long-horizon tasks.

You will receive a failed execution trace from an agent task.
Your job is to classify the primary root cause of failure into exactly one label from this closed set:

1. Environment Disturbance / Unable to Detect Change
Definition: The environment changes or rejects an action, but the agent fails to notice or properly react.
Example: The agent misses a permission denial and continues issuing commands based on a false belief of success.

2. Instruction Error
Definition: The agent misunderstands or ignores explicit task instructions, constraints, or exception clauses.
Example: The agent applies a recursive command globally, ignoring specific exception clauses in the prompt.

3. Catastrophic Forgetting
Definition: The agent forgets an important earlier constraint, goal, or fact and later violates it in a serious way.
Example: The agent deletes a system file, forgetting an earlier constraint to not modify sensitive files.

4. False Assumptions
Definition: The agent acts on an unverified belief about the environment, files, users, permissions, or task state.
Example: The agent assumes certain files are owned by the system user without actually verifying ownership.

5. Planning Errors (Sub-plan & Action)
Definition: The agent forms an incorrect sub-plan or chooses an inappropriate action sequence even if it understood the instruction.
Example: Applying a blanket permission change instead of first identifying the eligible target files.

6. History Error Accumulation
Definition: An earlier mistake propagates because the agent treats a failed or invalid earlier step as if it succeeded, causing downstream errors.
Example: An invalid command fails silently, but the agent assumes success and issues dependent commands, cascading the error.

7. Memory Limitations
Definition: The agent loses access to earlier relevant context because of context-window or recall limitations.
Example: Important earlier notices about file ownership are truncated from the context window.

Rules:
- Return exactly one label from the closed set above.
- Choose the single best primary root cause, not multiple labels.
- Prefer the earliest root cause that best explains the downstream failure.
- If the agent simply misunderstood the task requirements, choose Instruction Error.
- If the agent understood the task but picked a bad method or ordering, choose Planning Errors (Sub-plan & Action).
- If the agent explicitly relies on something it never checked, choose False Assumptions.
- If the agent forgets an earlier instruction or fact that was previously available in the trace, choose Catastrophic Forgetting.
- If the trace suggests the needed earlier information dropped out of context or was no longer accessible, choose Memory Limitations.
- If an earlier failed step causes later dependent mistakes because the agent assumes success, choose History Error Accumulation.
- If the environment gives feedback like permission denied, missing file, changed state, or other disturbance and the agent fails to detect/adapt, choose Environment Disturbance / Unable to Detect Change.

Output format:
Return only the label text, with no explanation, no JSON, and no extra words.
"""


def build_user_prompt(trace_text: str) -> str:
    return f"""Classify the following failed execution trace into exactly one failure label.

Failed execution trace:
<<<TRACE
{trace_text}
TRACE>>>"""


def normalize_label(raw_label: str) -> str:
    cleaned = " ".join((raw_label or "").strip().split())
    lowered = cleaned.lower()

    alias_map = {
        "environment disturbance / unable to detect change": FAILURE_LABELS[0],
        "environment disturbance": FAILURE_LABELS[0],
        "unable to detect change": FAILURE_LABELS[0],
        "instruction error": FAILURE_LABELS[1],
        "catastrophic forgetting": FAILURE_LABELS[2],
        "false assumptions": FAILURE_LABELS[3],
        "planning errors (sub-plan & action)": FAILURE_LABELS[4],
        "planning error": FAILURE_LABELS[4],
        "planning errors": FAILURE_LABELS[4],
        "sub-plan & action": FAILURE_LABELS[4],
        "history error accumulation": FAILURE_LABELS[5],
        "memory limitations": FAILURE_LABELS[6],
    }

    if lowered in alias_map:
        return alias_map[lowered]

    for label in FAILURE_LABELS:
        if label.lower() in lowered:
            return label

    raise ValueError(f"Model returned an invalid label: {raw_label!r}")


def classify_trace(
    client: OpenAI,
    model: str,
    trace_text: str,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(trace_text)},
                ],
                stream=False,
                temperature=0,
            )
            raw_label = response.choices[0].message.content
            return normalize_label(raw_label)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to classify trace after {max_retries} attempts: {last_error}")


def resolve_trace_column(df: pd.DataFrame, trace_column: Optional[str]) -> str:
    if trace_column:
        if trace_column not in df.columns:
            raise ValueError(
                f"Column {trace_column!r} not found. Available columns: {list(df.columns)}"
            )
        return trace_column

    if len(df.columns) == 1:
        return df.columns[0]

    raise ValueError(
        "CSV has multiple columns. Please provide --trace-column with the column name "
        "that contains the failed execution traces."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify failed execution traces in a CSV using DeepSeek and write labels back to a new column."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "--trace-column",
        help="Name of the column containing failed execution traces. If omitted, the script uses the only column in the CSV.",
    )
    parser.add_argument(
        "--output-column",
        default="failure_type",
        help="Name of the new or updated column for predicted labels. Default: failure_type",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path for the output CSV. If omitted, the input CSV is updated in place.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model name. Default: deepseek-chat",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek API base URL. Default: https://api.deepseek.com",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between API calls. Default: 0",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set in the environment.")

    df = pd.read_csv(args.csv_path)
    trace_column = resolve_trace_column(df, args.trace_column)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    predictions = []
    total_rows = len(df)

    for idx, trace in enumerate(df[trace_column].tolist(), start=1):
        if pd.isna(trace) or not str(trace).strip():
            predictions.append("")
            print(f"[{idx}/{total_rows}] Empty trace. Skipped.", file=sys.stderr)
            continue

        label = classify_trace(client=client, model=args.model, trace_text=str(trace))
        predictions.append(label)
        print(f"[{idx}/{total_rows}] {label}", file=sys.stderr)

        if args.delay > 0:
            time.sleep(args.delay)

    df[args.output_column] = predictions

    output_path = args.output_path or args.csv_path
    df.to_csv(output_path, index=False)
    print(f"Saved classified CSV to: {output_path}")


if __name__ == "__main__":
    main()