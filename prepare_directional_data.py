import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "processed" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Data" / "processed" / "directional"
SPLIT_NAMES = ("train", "validation", "test")
SOURCE_FIELDS = {"id", "dialect", "dialect_text", "standard_text"}
OUTPUT_FIELDS = {
    "example_id",
    "source_id",
    "dialect",
    "direction",
    "input_text",
    "target_text",
}
DIALECTS = {"D3", "D5"}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_split(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
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
            if not isinstance(record, dict) or not SOURCE_FIELDS.issubset(record):
                raise ValueError(
                    f"{path}:{line_number} must contain id, dialect, "
                    "dialect_text, and standard_text."
                )

            for field in SOURCE_FIELDS:
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
                    f"{path}:{line_number} has duplicate source ID {record['id']}."
                )

            seen_ids.add(record["id"])
            records.append(record)

    return records


def make_examples(record: dict[str, str]) -> list[dict[str, str]]:
    dialect = record["dialect"]
    directions = (
        (
            f"{dialect}_TO_STANDARD",
            record["dialect_text"],
            record["standard_text"],
        ),
        (
            f"STANDARD_TO_{dialect}",
            record["standard_text"],
            record["dialect_text"],
        ),
    )
    examples = []
    for direction, input_text, target_text in directions:
        examples.append(
            {
                "example_id": f"{record['id']}::{direction}",
                "source_id": record["id"],
                "dialect": dialect,
                "direction": direction,
                "input_text": f"{direction}: {input_text}",
                "target_text": target_text,
            }
        )
    return examples


def validate_splits(
    source_splits: dict[str, list[dict[str, str]]],
    example_splits: dict[str, list[dict[str, str]]],
) -> None:
    source_ids_by_split = {
        split: {record["id"] for record in records}
        for split, records in source_splits.items()
    }
    for index, split in enumerate(SPLIT_NAMES):
        for other_split in SPLIT_NAMES[index + 1 :]:
            overlap = source_ids_by_split[split] & source_ids_by_split[other_split]
            if overlap:
                raise ValueError(
                    f"Source IDs overlap between {split} and {other_split}: "
                    f"{sorted(overlap)[:3]}."
                )

            texts_by_split = {
                name: {
                    normalize_text(record[field])
                    for record in source_splits[name]
                    for field in ("dialect_text", "standard_text")
                }
                for name in (split, other_split)
            }
            text_overlap = texts_by_split[split] & texts_by_split[other_split]
            if text_overlap:
                raise ValueError(
                    f"Normalized sentence text overlaps between {split} and "
                    f"{other_split}."
                )

    seen_example_ids: set[str] = set()
    for split in SPLIT_NAMES:
        examples = example_splits[split]
        expected_count = len(source_splits[split]) * 2
        if len(examples) != expected_count:
            raise ValueError(
                f"{split} has {len(examples)} directional examples; "
                f"expected {expected_count}."
            )

        expected_source_ids = source_ids_by_split[split]
        actual_source_ids = set()
        directions_by_source: dict[str, set[str]] = {}
        for line_number, example in enumerate(examples, start=1):
            if set(example) != OUTPUT_FIELDS:
                raise ValueError(
                    f"{split} generated example {line_number} has invalid fields."
                )
            for field in OUTPUT_FIELDS:
                if not isinstance(example[field], str) or not example[field].strip():
                    raise ValueError(
                        f"{split} generated example {line_number} has an "
                        f"empty or invalid {field}."
                    )
            if example["example_id"] in seen_example_ids:
                raise ValueError(
                    f"Duplicate generated example ID {example['example_id']}."
                )
            seen_example_ids.add(example["example_id"])
            actual_source_ids.add(example["source_id"])

            dialect = example["dialect"]
            if dialect not in DIALECTS:
                raise ValueError(
                    f"{split} generated example has unsupported dialect {dialect!r}."
                )
            allowed_directions = {
                f"{dialect}_TO_STANDARD",
                f"STANDARD_TO_{dialect}",
            }
            if example["direction"] not in allowed_directions:
                raise ValueError(
                    f"{split} generated example has invalid direction "
                    f"{example['direction']!r} for {dialect}."
                )
            if not example["input_text"].startswith(f"{example['direction']}: "):
                raise ValueError(
                    f"{split} generated input is missing its direction prefix."
                )
            directions_by_source.setdefault(example["source_id"], set()).add(
                example["direction"]
            )

        if actual_source_ids != expected_source_ids:
            raise ValueError(
                f"Generated {split} examples do not map exactly to that split's "
                "source IDs."
            )
        for source_record in source_splits[split]:
            expected_directions = {
                f"{source_record['dialect']}_TO_STANDARD",
                f"STANDARD_TO_{source_record['dialect']}",
            }
            if directions_by_source.get(source_record["id"]) != expected_directions:
                raise ValueError(
                    f"Source ID {source_record['id']} does not have exactly its "
                    "two expected directions."
                )


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and validate directional examples for each data split."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_splits = {
        split: read_split(args.input_dir / f"{split}.jsonl")
        for split in SPLIT_NAMES
    }
    # Convert within each split so validation and test records stay isolated.
    example_splits = {
        split: [
            example
            for record in source_splits[split]
            for example in make_examples(record)
        ]
        for split in SPLIT_NAMES
    }
    validate_splits(source_splits, example_splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        write_jsonl(args.output_dir / f"{split}.jsonl", example_splits[split])

    print("Directional data prepared and validated:")
    for split in SPLIT_NAMES:
        direction_counts = Counter(
            example["direction"] for example in example_splits[split]
        )
        print(
            f"{split}: {len(source_splits[split])} source records -> "
            f"{len(example_splits[split])} examples "
            f"({dict(sorted(direction_counts.items()))})"
        )
        print(f"  Created: {args.output_dir / f'{split}.jsonl'}")
    print(
        "Validation passed: required fields, non-empty values, unique example IDs, "
        "two directions per source, and split isolation."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
