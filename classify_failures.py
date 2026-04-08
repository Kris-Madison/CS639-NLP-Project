import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

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

You will receive:
- the original task description
- one failed execution trace from the agent

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
- Use both the task description and the trace.
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


def build_user_prompt(task_description: str, trace_text: str) -> str:
    return f"""Classify the failed agent run into exactly one failure label.

Task description:
<<<TASK
{task_description}
TASK>>>

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

    raise ValueError(f"Invalid label returned by model: {raw_label!r}")


def classify_trace(
    client: OpenAI,
    model: str,
    task_description: str,
    trace_text: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(task_description, trace_text),
                    },
                ],
                temperature=0,
                stream=False,
            )
            raw_label = response.choices[0].message.content
            return normalize_label(raw_label)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def parse_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

    return records


def extract_task_description(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        function_args = function.get("arguments", "")

        lines.append(f"Assistant tool call: {function_name}")
        if function_args:
            lines.append(f"Function arguments: {function_args}")

    return lines


def format_trace(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for message in messages:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()

        if role == "user":
            lines.append("User message:")
            lines.append(content or "[empty]")

        elif role == "assistant":
            if content:
                lines.append("Assistant content:")
                lines.append(content)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                lines.extend(format_tool_calls(tool_calls))

        elif role == "tool":
            lines.append("Tool output:")
            lines.append(content or "[empty]")

            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                lines.append(f"Tool call id: {tool_call_id}")

        else:
            lines.append(f"{role.capitalize()} message:")
            lines.append(content or "[empty]")

        lines.append("")

    return "\n".join(lines).strip()


def sample_one_failed_run_per_task(
    records: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    failures_by_task: Dict[Any, List[Dict[str, Any]]] = {}

    for record in records:
        if record.get("result") != 0:
            continue

        task_index = record.get("task_index")
        if task_index is None:
            continue

        failures_by_task.setdefault(task_index, []).append(record)

    rng = random.Random(seed)
    sampled_records: List[Dict[str, Any]] = []

    for task_index in sorted(failures_by_task):
        sampled_records.append(rng.choice(failures_by_task[task_index]))

    return sampled_records


def save_results(rows: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_index", "classification_type"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one random failed run per task from a JSONL file, "
            "classify each failure with DeepSeek, and save task_index plus "
            "classification_type to a CSV file."
        )
    )
    parser.add_argument("jsonl_path", help="Path to the input JSONL file.")
    parser.add_argument(
        "--output-path",
        default="failure_classifications.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between API calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling one failed run per task.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set.")

    records = parse_jsonl(args.jsonl_path)
    sampled_failures = sample_one_failed_run_per_task(records, seed=args.seed)

    if not sampled_failures:
        raise ValueError("No failed runs with result == 0 were found.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    output_rows: List[Dict[str, Any]] = []
    total = len(sampled_failures)

    for index, record in enumerate(sampled_failures, start=1):
        task_index = record["task_index"]
        messages = record.get("messages", [])

        task_description = extract_task_description(messages)
        trace_text = format_trace(messages)

        label = classify_trace(
            client=client,
            model=args.model,
            task_description=task_description,
            trace_text=trace_text,
        )

        output_rows.append(
            {
                "task_index": task_index,
                "classification_type": label,
            }
        )

        print(f"[{index}/{total}] task_index={task_index} -> {label}", file=sys.stderr)

        if args.delay > 0:
            time.sleep(args.delay)

    save_results(output_rows, args.output_path)
    print(f"Saved classifications to: {args.output_path}")


if __name__ == "__main__":
    main()
