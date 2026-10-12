import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Data" / "IISc_RESPIN_train_kn_small"
OUTPUT_ROOT = PROJECT_ROOT / "Data" / "processed"
SAMPLES_PER_DIALECT = 500
RANDOM_SEED = 42


def read_entries(dialect: str) -> tuple[list[dict[str, str]], int, int]:
    dialect_root = DATASET_ROOT / dialect
    if not dialect_root.is_dir():
        raise FileNotFoundError(f"Dialect directory not found: {dialect_root}")

    entries: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    malformed_lines = 0
    duplicate_ids = 0

    for transcript_path in sorted(dialect_root.rglob("*.txt")):
        with transcript_path.open(encoding="utf-8") as transcript_file:
            for line in transcript_file:
                if not line.strip():
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    malformed_lines += 1
                    continue

                audio_id, text = parts[0].strip(), parts[1].strip()
                if not audio_id or not text:
                    malformed_lines += 1
                    continue
                if audio_id in seen_ids:
                    duplicate_ids += 1
                    continue

                seen_ids.add(audio_id)
                entries.append({"id": audio_id, "dialect": dialect, "text": text})

    return entries, malformed_lines, duplicate_ids


def select_samples(
    dialect: str, rng: random.Random, selected_ids: set[str]
) -> tuple[list[dict[str, str]], int, int]:
    entries, malformed_lines, duplicate_ids = read_entries(dialect)
    rng.shuffle(entries)

    selected: list[dict[str, str]] = []
    for entry in entries:
        if entry["id"] in selected_ids:
            continue
        selected.append(entry)
        selected_ids.add(entry["id"])
        if len(selected) == SAMPLES_PER_DIALECT:
            break

    if len(selected) != SAMPLES_PER_DIALECT:
        raise ValueError(
            f"{dialect} has only {len(selected)} valid unique entries; "
            f"{SAMPLES_PER_DIALECT} are required."
        )

    return selected, malformed_lines, duplicate_ids


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_outputs(
    output_paths: dict[str, Path], combined_path: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dialect_records: dict[str, list[dict[str, str]]] = {}
    for dialect, path in output_paths.items():
        with path.open(encoding="utf-8") as input_file:
            records = [json.loads(line) for line in input_file if line.strip()]
        if len(records) != SAMPLES_PER_DIALECT:
            raise ValueError(f"{path} contains {len(records)} records, expected 500.")
        if any(record.get("dialect") != dialect for record in records):
            raise ValueError(f"{path} contains an unexpected dialect value.")
        dialect_records[dialect] = records

    with combined_path.open(encoding="utf-8") as input_file:
        combined_records = [json.loads(line) for line in input_file if line.strip()]

    expected_records = dialect_records["D3"] + dialect_records["D5"]
    if len(combined_records) != 2 * SAMPLES_PER_DIALECT:
        raise ValueError(
            f"{combined_path} contains {len(combined_records)} records, expected 1000."
        )
    if combined_records != expected_records:
        raise ValueError(f"{combined_path} does not match the dialect output files.")

    ids = [record.get("id") for record in combined_records]
    if len(set(ids)) != len(ids):
        raise ValueError("The selected records contain duplicate IDs.")
    if any(not isinstance(record.get("text"), str) or not record["text"].strip()
           for record in combined_records):
        raise ValueError("The selected records contain empty text.")
    if any(set(record) != {"id", "dialect", "text"} for record in combined_records):
        raise ValueError("A JSONL record does not have exactly id, dialect, and text.")

    return dialect_records["D3"], dialect_records["D5"]


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    selected_ids: set[str] = set()
    selected_by_dialect: dict[str, list[dict[str, str]]] = {}
    input_stats: dict[str, tuple[int, int]] = {}

    for dialect in ("D3", "D5"):
        records, malformed_lines, duplicate_ids = select_samples(
            dialect, rng, selected_ids
        )
        selected_by_dialect[dialect] = records
        input_stats[dialect] = malformed_lines, duplicate_ids

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_paths = {
        dialect: OUTPUT_ROOT / f"{dialect.lower()}_500.jsonl"
        for dialect in ("D3", "D5")
    }
    combined_path = OUTPUT_ROOT / "combined_1000.jsonl"

    for dialect, path in output_paths.items():
        write_jsonl(path, selected_by_dialect[dialect])
    write_jsonl(
        combined_path,
        selected_by_dialect["D3"] + selected_by_dialect["D5"],
    )

    d3_records, d5_records = validate_outputs(output_paths, combined_path)

    print(f"Extraction complete (random seed: {RANDOM_SEED})")
    for dialect, records in (("D3", d3_records), ("D5", d5_records)):
        malformed_lines, duplicate_ids = input_stats[dialect]
        print(
            f"{dialect}: {len(records)} records; "
            f"{malformed_lines} malformed non-empty lines skipped; "
            f"{duplicate_ids} duplicate IDs skipped"
        )
    print(f"Combined: {len(d3_records) + len(d5_records)} records")
    print("Validation passed: exact per-dialect counts, unique IDs, non-empty text, UTF-8 JSONL.")
    for path in (*output_paths.values(), combined_path):
        print(f"Created: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
