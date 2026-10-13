import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from standardize_kannada import standardize_one


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "Data" / "processed" / "splits" / "test.jsonl"
DIALECTS = ("D3", "D5")
SENTENCES_PER_DIALECT = 5
MIN_REQUEST_INTERVAL_SECONDS = 5.0
REQUIRED_FIELDS = {"id", "dialect", "dialect_text", "standard_text"}


def read_examples(path: Path, seed: int) -> list[dict[str, str]]:
    by_dialect: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON: {error}"
                ) from error
            if not isinstance(record, dict) or set(record) != REQUIRED_FIELDS:
                raise ValueError(
                    f"{path}:{line_number} must contain exactly "
                    "id, dialect, dialect_text, and standard_text."
                )
            for field in REQUIRED_FIELDS:
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(
                        f"{path}:{line_number} has an empty or invalid {field}."
                    )
            if record["dialect"] not in DIALECTS:
                raise ValueError(
                    f"{path}:{line_number} has unsupported dialect "
                    f"{record['dialect']!r}."
                )
            if record["id"] in seen_ids:
                raise ValueError(
                    f"{path}:{line_number} has duplicate ID {record['id']}."
                )

            seen_ids.add(record["id"])
            by_dialect[record["dialect"]].append(record)

    rng = random.Random(seed)
    selected = []
    for dialect in DIALECTS:
        available = by_dialect[dialect]
        if len(available) < SENTENCES_PER_DIALECT:
            raise ValueError(
                f"{path} has only {len(available)} {dialect} records; "
                f"{SENTENCES_PER_DIALECT} are required."
            )
        selected.extend(rng.sample(available, SENTENCES_PER_DIALECT))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live Gemini D3/D5-to-Standard-Kannada conversions and time them."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    examples = read_examples(args.input, args.seed)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to the Git-ignored project .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )
    successful_times: list[float] = []
    failures = 0
    started_at = time.perf_counter()
    last_request_started: float | None = None

    print(f"Live model: {args.model}")
    print(f"Examples: 5 D3 + 5 D5 (sample seed: {args.seed})")
    print("Each request is timed separately; requests are spaced at least 5 seconds apart.")

    try:
        for number, example in enumerate(examples, start=1):
            if last_request_started is not None:
                wait = (
                    MIN_REQUEST_INTERVAL_SECONDS
                    - (time.perf_counter() - last_request_started)
                )
                if wait > 0:
                    time.sleep(wait)

            print(
                f"\n[{number}/10] {example['dialect']} "
                f"(ID: {example['id']})"
            )
            print(f"Dialect input: {example['dialect_text']}")
            request_started = time.perf_counter()
            last_request_started = request_started
            try:
                converted = standardize_one(
                    client, args.model, example["dialect_text"]
                )
            except Exception as error:
                failures += 1
                message = str(error).replace(api_key, "[REDACTED]")
                print(
                    f"Conversion failed: {type(error).__name__}: {message}",
                    file=sys.stderr,
                )
                continue

            elapsed = time.perf_counter() - request_started
            successful_times.append(elapsed)
            print(f"LLM output: {converted}")
            print(f"Request time: {elapsed:.2f} seconds")
    finally:
        client.close()

    total_elapsed = time.perf_counter() - started_at
    print("\nTiming summary")
    print(f"Successful conversions: {len(successful_times)}/10")
    print(f"Failed conversions: {failures}")
    if successful_times:
        print(
            f"Total successful request latency: "
            f"{sum(successful_times):.2f} seconds"
        )
        print(
            f"Average client-observed latency per request: "
            f"{sum(successful_times) / len(successful_times):.2f} seconds"
        )
    print(
        f"Total demo time (includes rate-limit pauses): "
        f"{total_elapsed:.2f} seconds"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
