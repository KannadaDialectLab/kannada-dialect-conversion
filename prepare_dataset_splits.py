import argparse
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "Data" / "processed" / "standardized_1000.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Data" / "processed" / "splits"
EXPECTED_DIALECT_COUNTS = {"D3": 500, "D5": 500}
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
DISTRIBUTION_TOLERANCE = 0.02
RECORD_FIELDS = {"id", "dialect", "dialect_text", "standard_text"}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def read_dataset(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    dialect_counts = Counter()

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
            if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
                raise ValueError(
                    f"{path}:{line_number} must contain exactly "
                    "id, dialect, dialect_text, and standard_text."
                )

            for field in RECORD_FIELDS:
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(
                        f"{path}:{line_number} has an empty or invalid {field}."
                    )
            if record["dialect"] not in EXPECTED_DIALECT_COUNTS:
                raise ValueError(
                    f"{path}:{line_number} has unsupported dialect "
                    f"{record['dialect']!r}."
                )
            if record["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number} has duplicate ID {record['id']}.")

            seen_ids.add(record["id"])
            dialect_counts[record["dialect"]] += 1
            records.append(record)

    expected_total = sum(EXPECTED_DIALECT_COUNTS.values())
    if len(records) != expected_total:
        raise ValueError(f"{path} contains {len(records)} records; expected {expected_total}.")
    if dialect_counts != EXPECTED_DIALECT_COUNTS:
        raise ValueError(
            f"Expected dialect counts {EXPECTED_DIALECT_COUNTS}; found {dict(dialect_counts)}."
        )
    return records


def get_stratum(record: dict[str, str]) -> tuple[str, bool]:
    unchanged = normalize_text(record["dialect_text"]) == normalize_text(
        record["standard_text"]
    )
    return record["dialect"], unchanged


def duplicate_groups(records: list[dict[str, str]]) -> list[list[int]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    first_record_by_text: dict[str, int] = {}
    for index, record in enumerate(records):
        for field in ("dialect_text", "standard_text"):
            text = normalize_text(record[field])
            if text in first_record_by_text:
                union(index, first_record_by_text[text])
            else:
                first_record_by_text[text] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    return list(groups.values())


def split_records(
    records: list[dict[str, str]], seed: int
) -> dict[str, list[dict[str, str]]]:
    groups = duplicate_groups(records)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    strata = sorted({get_stratum(record) for record in records})
    total_strata = Counter(get_stratum(record) for record in records)
    targets = {
        split: {
            stratum: total_strata[stratum] * ratio
            for stratum in strata
        }
        for split, ratio in SPLIT_RATIOS.items()
    }
    expected_sizes = {
        split: len(records) * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    assigned_strata = {split: Counter() for split in SPLIT_RATIOS}
    assigned_sizes = Counter()
    record_splits: dict[int, str] = {}

    for group in groups:
        group_strata = Counter(get_stratum(records[index]) for index in group)

        def score(candidate: str) -> float:
            result = 0.0
            for split in SPLIT_RATIOS:
                for stratum in strata:
                    count = assigned_strata[split][stratum]
                    if split == candidate:
                        count += group_strata[stratum]
                    target = targets[split][stratum]
                    result += (count - target) ** 2 / max(target, 1.0)

                size = assigned_sizes[split]
                if split == candidate:
                    size += len(group)
                target_size = expected_sizes[split]
                result += (size - target_size) ** 2 / max(target_size, 1.0)
            return result

        destination = min(SPLIT_RATIOS, key=score)
        assigned_strata[destination].update(group_strata)
        assigned_sizes[destination] += len(group)
        for index in group:
            record_splits[index] = destination

    return {
        split: [
            record for index, record in enumerate(records)
            if record_splits[index] == split
        ]
        for split in SPLIT_RATIOS
    }


def validate_splits(
    records: list[dict[str, str]],
    splits: dict[str, list[dict[str, str]]],
) -> None:
    source_ids = {record["id"] for record in records}
    all_split_ids: set[str] = set()
    source_strata = Counter(get_stratum(record) for record in records)

    for split, split_records in splits.items():
        expected_size = int(len(records) * SPLIT_RATIOS[split])
        if len(split_records) != expected_size:
            raise ValueError(
                f"{split} has {len(split_records)} records; expected {expected_size}."
            )
        split_ids = {record["id"] for record in split_records}
        if len(split_ids) != len(split_records):
            raise ValueError(f"{split} contains duplicate IDs.")
        if all_split_ids.intersection(split_ids):
            raise ValueError(f"Record IDs overlap with {split}.")
        all_split_ids.update(split_ids)

        split_strata = Counter(get_stratum(record) for record in split_records)
        for stratum, source_count in source_strata.items():
            source_rate = source_count / len(records)
            split_rate = split_strata[stratum] / len(split_records)
            if abs(source_rate - split_rate) > DISTRIBUTION_TOLERANCE:
                raise ValueError(
                    f"{split} does not preserve the dialect/changed distribution "
                    f"for {stratum}: {split_rate:.1%} versus {source_rate:.1%}."
                )

    if all_split_ids != source_ids:
        raise ValueError("The splits do not contain exactly the source dataset IDs.")

    split_by_text: dict[str, str] = {}
    for split, split_records in splits.items():
        for record in split_records:
            for field in ("dialect_text", "standard_text"):
                text = normalize_text(record[field])
                previous_split = split_by_text.get(text)
                if previous_split is not None and previous_split != split:
                    raise ValueError(
                        f"Sentence leakage: identical text appears in "
                        f"{previous_split} and {split}."
                    )
                split_by_text[text] = split


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and validate reproducible, leakage-safe dataset splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_dataset(args.input)
    splits = split_records(records, args.seed)
    validate_splits(records, splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records_in_split in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", records_in_split)

    print(f"Validated splits (seed: {args.seed}; source left unchanged):")
    for split, records_in_split in splits.items():
        counts = Counter(get_stratum(record) for record in records_in_split)
        print(
            f"{split}: {len(records_in_split)} records; "
            f"D3 changed/unchanged {counts[('D3', False)]}/"
            f"{counts[('D3', True)]}; "
            f"D5 changed/unchanged {counts[('D5', False)]}/"
            f"{counts[('D5', True)]}"
        )
        print(f"  Created: {args.output_dir / f'{split}.jsonl'}")
    print("Validation passed: IDs, stratification, and cross-split sentence leakage.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
