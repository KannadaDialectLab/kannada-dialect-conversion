import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "Data" / "processed" / "combined_1000.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "processed" / "standardized_1000.jsonl"
DEFAULT_FAILURES = (
    PROJECT_ROOT / "Data" / "processed" / "standardized_1000.failures.jsonl"
)
EXPECTED_COUNTS = {"D3": 500, "D5": 500}
MIN_REQUEST_INTERVAL_SECONDS = 5.0
KANNADA_RANGE = range(0x0C80, 0x0D00)
MOJIBAKE_PATTERN = re.compile(r"à²|Ã|â")
STANDARDIZATION_PROMPT = """You are a Kannada language standardization system.

Convert the following Kannada dialect sentence into natural Standard Kannada.

Preserve the exact meaning. Do not add or remove information. Do not translate
to English. Return only the Standard Kannada sentence, in Kannada script.
Preserve names, numbers, abbreviations, technical terms, and entities where
appropriate. Do not transliterate into Latin script or unnecessarily rewrite
the sentence.

Dialect sentence:
{sentence}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            records.append(record)
    return records


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_inputs(path: Path) -> list[dict[str, str]]:
    records = read_jsonl(path)
    if len(records) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(
            f"{path} contains {len(records)} records; expected 1000."
        )

    seen_ids: set[str] = set()
    counts = {dialect: 0 for dialect in EXPECTED_COUNTS}
    for line_number, record in enumerate(records, start=1):
        if set(record) != {"id", "dialect", "text"}:
            raise ValueError(
                f"{path}:{line_number} must have exactly id, dialect, and text."
            )
        record_id = record["id"]
        dialect = record["dialect"]
        text = record["text"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"{path}:{line_number} has an empty or invalid id.")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate input ID: {record_id}")
        seen_ids.add(record_id)
        if dialect not in EXPECTED_COUNTS:
            raise ValueError(f"{path}:{line_number} has unsupported dialect {dialect!r}.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number} has empty or invalid text.")
        counts[dialect] += 1

    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected 500 D3 and 500 D5 records; found {counts}.")
    return records


def has_kannada(text: str) -> bool:
    return any(ord(character) in KANNADA_RANGE for character in text)


def validate_standard_text(text: str) -> str:
    standardized = text.strip()
    if not standardized:
        raise ValueError("Gemini returned an empty response.")
    if not has_kannada(standardized):
        raise ValueError("Gemini response does not contain Kannada Unicode text.")
    if MOJIBAKE_PATTERN.search(standardized):
        raise ValueError("Gemini response appears to contain mojibake.")
    return standardized


def load_completed(
    path: Path, inputs_by_id: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    completed: dict[str, dict[str, str]] = {}
    if not path.exists():
        return completed
    for line_number, record in enumerate(read_jsonl(path), start=1):
        if set(record) != {"id", "dialect", "dialect_text", "standard_text"}:
            raise ValueError(
                f"{path}:{line_number} must have exactly id, dialect, "
                "dialect_text, and standard_text."
            )
        record_id = record["id"]
        if not isinstance(record_id, str) or record_id not in inputs_by_id:
            raise ValueError(f"{path}:{line_number} has an ID absent from the input.")
        if record_id in completed:
            raise ValueError(f"{path} contains duplicate ID {record_id}.")
        source = inputs_by_id[record_id]
        if (
            record["dialect"] != source["dialect"]
            or record["dialect_text"] != source["text"]
        ):
            raise ValueError(
                f"{path}:{line_number} changed input fields for ID {record_id}."
            )
        if not isinstance(record["standard_text"], str):
            raise ValueError(f"{path}:{line_number} has invalid standard_text.")
        validate_standard_text(record["standard_text"])
        completed[record_id] = record
    return completed


def load_failures(
    path: Path, inputs_by_id: dict[str, dict[str, str]]
) -> dict[str, dict[str, Any]]:
    failures: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return failures
    for line_number, record in enumerate(read_jsonl(path), start=1):
        if set(record) != {"id", "dialect", "dialect_text", "attempts", "error"}:
            raise ValueError(
                f"{path}:{line_number} must have exactly id, dialect, "
                "dialect_text, attempts, and error."
            )
        record_id = record["id"]
        if not isinstance(record_id, str) or record_id not in inputs_by_id:
            raise ValueError(f"{path}:{line_number} has an ID absent from the input.")
        if record_id in failures:
            raise ValueError(f"{path} contains duplicate ID {record_id}.")
        source = inputs_by_id[record_id]
        if (
            record["dialect"] != source["dialect"]
            or record["dialect_text"] != source["text"]
        ):
            raise ValueError(
                f"{path}:{line_number} changed input fields for ID {record_id}."
            )
        failures[record_id] = record
    return failures


def safe_error_message(error: Exception, api_key: str) -> str:
    message = str(error).replace(api_key, "[REDACTED]")
    return f"{type(error).__name__}: {message}"


def retry_delay_seconds(error: Exception, attempt: int) -> float | None:
    error_details: list[str] = []
    current_error: BaseException | None = error
    seen_errors: set[int] = set()
    while current_error is not None and id(current_error) not in seen_errors:
        seen_errors.add(id(current_error))
        error_details.append(
            f"{type(current_error).__name__}: {current_error}"
        )
        current_error = current_error.__cause__ or current_error.__context__
    message = "\n".join(error_details)
    status_code = getattr(error, "code", None) or getattr(
        error, "status_code", None
    )
    if status_code is None:
        match = re.search(r"\b(429|503|504)\b", message)
        status_code = int(match.group(1)) if match else None
    else:
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None
    if status_code is None:
        match = re.search(r"\b(429|503|504)\b", message)
        status_code = int(match.group(1)) if match else None

    network_error_names = (
        "SSLCertVerificationError",
        "gaierror",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "RemoteProtocolError",
        "NetworkError",
        "TransportError",
    )
    network_error_markers = (
        "CERTIFICATE_VERIFY_FAILED",
        "getaddrinfo failed",
        "temporary failure in name resolution",
        "name or service not known",
        "connection reset",
        "connection refused",
        "network is unreachable",
    )
    is_network_error = any(name.lower() in message.lower() for name in network_error_names)
    is_network_error = is_network_error or any(
        marker.lower() in message.lower() for marker in network_error_markers
    )
    if status_code not in {429, 503, 504} and not is_network_error:
        return None

    delay = min(
        MIN_REQUEST_INTERVAL_SECONDS * (2 ** (attempt - 1)),
        300.0,
    )
    retry_matches = re.findall(
        r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        message,
        re.IGNORECASE,
    )
    retry_matches.extend(
        re.findall(
            r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"',
            message,
            re.IGNORECASE,
        )
    )
    if retry_matches:
        delay = max(delay, *(float(value) for value in retry_matches))
    return delay


def standardize_one(
    client: genai.Client, model: str, sentence: str
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=STANDARDIZATION_PROMPT.format(sentence=sentence),
        config=types.GenerateContentConfig(temperature=0),
    )
    if response.text is None:
        raise ValueError("Gemini returned no text.")
    return validate_standard_text(response.text)


def validate_and_report(
    inputs: list[dict[str, str]],
    completed: dict[str, dict[str, str]],
    failures: dict[str, dict[str, Any]],
    require_complete: bool,
) -> bool:
    input_ids = {record["id"] for record in inputs}
    inputs_by_id = {record["id"]: record for record in inputs}
    duplicate_ids = len(completed) - len(set(completed))
    empty_outputs = sum(
        not record["standard_text"].strip() for record in completed.values()
    )
    invalid_unicode = sum(
        not has_kannada(record["standard_text"])
        or bool(MOJIBAKE_PATTERN.search(record["standard_text"]))
        for record in completed.values()
    )
    mismatched_inputs = sum(
        record["dialect"] != inputs_by_id[record_id]["dialect"]
        or record["dialect_text"] != inputs_by_id[record_id]["text"]
        for record_id, record in completed.items()
    )
    complete = set(completed) == input_ids
    counts = {
        dialect: sum(record["dialect"] == dialect for record in inputs)
        for dialect in EXPECTED_COUNTS
    }

    print(f"Input records: {len(inputs)}")
    print(f"D3: {counts['D3']}")
    print(f"D5: {counts['D5']}")
    print(f"Successfully standardized: {len(completed)}")
    print(f"Failed: {len(failures)}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Empty outputs: {empty_outputs}")
    print(f"Invalid Kannada Unicode outputs: {invalid_unicode}")
    print(f"Changed/mismatched original fields: {mismatched_inputs}")
    if failures:
        print("Failed IDs:")
        for record_id, failure in failures.items():
            print(f"  {record_id}: {failure['error']}")
    if not complete:
        print(f"Remaining: {len(input_ids - set(completed))}")

    return (
        duplicate_ids == 0
        and empty_outputs == 0
        and invalid_unicode == 0
        and mismatched_inputs == 0
        and not failures
        and (complete or not require_complete)
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def minimum_delay_float(value: str) -> float:
    parsed = float(value)
    if parsed < MIN_REQUEST_INTERVAL_SECONDS:
        raise argparse.ArgumentTypeError(
            "must be at least 5 seconds to stay below the API rate limit"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize the local D3/D5 JSONL pilot dataset with Gemini. "
            "Successful records are checkpointed for resume."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="only consider the first N input records (for example, --limit 5)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=minimum_delay_float,
        default=MIN_REQUEST_INTERVAL_SECONDS,
        help="minimum delay between API requests (at least 5 seconds; default: 5)",
    )
    parser.add_argument(
        "--max-retries",
        type=positive_int,
        default=3,
        help="maximum attempts per record, including the first attempt (default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    failures_path = args.failures.resolve()
    if len({input_path, output_path, failures_path}) != 3:
        raise ValueError("Input, output, and failures paths must be different.")

    inputs = load_inputs(input_path)
    inputs_by_id = {record["id"]: record for record in inputs}
    completed = load_completed(output_path, inputs_by_id)
    failures = load_failures(failures_path, inputs_by_id)
    for record_id in completed:
        failures.pop(record_id, None)
    if failures_path.exists():
        write_jsonl_atomic(
            failures_path,
            [
                failures[item["id"]]
                for item in inputs
                if item["id"] in failures
            ],
        )

    selected = inputs[: args.limit] if args.limit is not None else inputs
    pending = [record for record in selected if record["id"] not in completed]
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if pending and not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in the environment or in a "
            "Git-ignored .env file."
        )

    client = (
        genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60_000),
        )
        if pending
        else None
    )
    last_request_at: float | None = None
    next_request_not_before = 0.0
    try:
        for source in pending:
            record_id = source["id"]
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                now = time.monotonic()
                earliest_request_at = max(
                    next_request_not_before,
                    (
                        last_request_at + args.delay_seconds
                        if last_request_at is not None
                        else now
                    ),
                )
                wait_seconds = max(0.0, earliest_request_at - now)
                if wait_seconds:
                    time.sleep(wait_seconds)
                last_request_at = time.monotonic()
                try:
                    if client is None:
                        raise RuntimeError("Gemini client was not initialized.")
                    standard_text = standardize_one(
                        client, args.model, source["text"]
                    )
                    completed[record_id] = {
                        "id": record_id,
                        "dialect": source["dialect"],
                        "dialect_text": source["text"],
                        "standard_text": standard_text,
                    }
                    failures.pop(record_id, None)
                    write_jsonl_atomic(
                        output_path,
                        [
                            completed[item["id"]]
                            for item in inputs
                            if item["id"] in completed
                        ],
                    )
                    write_jsonl_atomic(
                        failures_path,
                        [
                            failures[item["id"]]
                            for item in inputs
                            if item["id"] in failures
                        ],
                    )
                    print(f"Saved {record_id}")
                    break
                except Exception as error:
                    last_error = error
                    backoff_seconds = retry_delay_seconds(error, attempt)
                    if backoff_seconds is not None and attempt < args.max_retries:
                        next_request_not_before = max(
                            next_request_not_before,
                            time.monotonic() + backoff_seconds,
                        )
                        print(
                            f"Retrying {record_id} after "
                            f"{backoff_seconds:.1f}s backoff "
                            f"(attempt {attempt + 1}/{args.max_retries})",
                            file=sys.stderr,
                        )
            else:
                assert last_error is not None
                failures[record_id] = {
                    "id": record_id,
                    "dialect": source["dialect"],
                    "dialect_text": source["text"],
                    "attempts": args.max_retries,
                    "error": safe_error_message(last_error, api_key),
                }
                write_jsonl_atomic(
                    failures_path,
                    [
                        failures[item["id"]]
                        for item in inputs
                        if item["id"] in failures
                    ],
                )
                print(
                    f"Failed {record_id} after {args.max_retries} attempts: "
                    f"{failures[record_id]['error']}",
                    file=sys.stderr,
                )
    finally:
        if client is not None:
            client.close()

    return (
        0
        if validate_and_report(
            inputs,
            completed,
            failures,
            require_complete=args.limit is None,
        )
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
