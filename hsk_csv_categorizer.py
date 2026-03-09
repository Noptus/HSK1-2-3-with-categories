#!/usr/bin/env python3
"""Classify HSK CSV rows into Category + Subcategory tags using OpenAI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

DEFAULT_INPUT_PATH = Path("data/.hsk1-2-3_vocab_original.csv")
DEFAULT_OUTPUT_PATH = Path("data/hsk1-2-3_vocab_categorized.csv")
DEFAULT_CHECKPOINT_PATH = Path(".cache/hsk1-2-3_vocab_categorizer.checkpoint.json")


TAXONOMY: Dict[str, List[str]] = {
    "Grammar & Function Words": [
        "addition",
        "contrast",
        "sequence",
        "cause_reason",
        "condition",
        "concession",
        "comparison",
        "question_marker",
        "aspect_marker",
        "negation",
        "structural_particle",
        "measure_word",
        "preposition_coverb",
    ],
    "People & Relationships": [
        "family",
        "social_role",
        "profession",
        "personal_identity",
    ],
    "Home & Daily Life": [
        "household_item",
        "daily_routine",
        "shopping_service",
        "clothing_personal_use",
    ],
    "Food & Dining": [
        "food_item",
        "drink_item",
        "dining_action",
        "taste_cooking",
    ],
    "Study & Work": [
        "school_learning",
        "workplace_tasks",
        "business_finance",
        "tools_materials",
    ],
    "Time & Events": [
        "calendar_time",
        "duration_frequency",
        "past_present_future",
        "milestone_event",
    ],
    "Numbers & Measurement": [
        "cardinal_number",
        "ordinal_number",
        "quantity_amount",
        "unit_measure",
    ],
    "Places & Transport": [
        "location_place",
        "direction_position",
        "transport_travel",
        "nature_environment",
    ],
    "Actions & Communication": [
        "physical_action",
        "motion_transfer",
        "speech_communication",
        "mental_process",
    ],
    "Qualities & States": [
        "physical_property",
        "evaluation_judgment",
        "emotion_attitude",
        "state_change",
    ],
}

SUBCATEGORY_TO_CATEGORY: Dict[str, str] = {}
for _category, _tags in TAXONOMY.items():
    for _tag in _tags:
        if _tag in SUBCATEGORY_TO_CATEGORY and SUBCATEGORY_TO_CATEGORY[_tag] != _category:
            raise ValueError(f"Subcategory tag mapped to multiple categories: {_tag}")
        SUBCATEGORY_TO_CATEGORY[_tag] = _category


@dataclass(frozen=True)
class Prediction:
    row_id: int
    category: str
    subcategories: List[str]
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Category and Subcategory columns to a CSV using OpenAI classification."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to inferred categorized filename.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSON path. Defaults to .cache/<input>.checkpoint.json",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional .env file for OPENAI_API_KEY / OPENAIAPI_KEY.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name (stronger affordable default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Rows per API request in pass 1.",
    )
    parser.add_argument(
        "--retry-batch-size",
        type=int,
        default=20,
        help="Rows per API request in retry passes.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry rounds for invalid/low-confidence rows.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.72,
        help="Rows below this confidence are retried.",
    )
    parser.add_argument(
        "--rescue-min-confidence",
        type=float,
        default=0.55,
        help=(
            "Final fallback threshold for unresolved rows after retries. "
            "Must be <= min-confidence."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: classify only first N rows (useful for low-cost checks).",
    )
    parser.add_argument(
        "--api-attempts",
        type=int,
        default=4,
        help="Attempts per API call before failing the batch.",
    )
    parser.add_argument(
        "--api-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff between failed API attempts.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("'").strip('"')
        if key and value and key not in os.environ:
            os.environ[key] = value


def get_api_key(env_path: Path) -> str:
    direct = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI_KEY")
    if direct:
        return direct
    load_env_file(env_path)
    file_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI_KEY")
    if file_key:
        return file_key
    raise RuntimeError(
        "Missing API key. Set OPENAI_API_KEY (or OPENAIAPI_KEY), "
        "or provide it in the env file."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category(raw: Any) -> str:
    text = str(raw or "").strip()
    if text in TAXONOMY:
        return text
    lowered = text.lower()
    for category in TAXONOMY:
        if category.lower() == lowered:
            return category
    return text


def normalize_subcategory(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")
    text = text.replace("/", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_subcategories(raw_values: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    for value in raw_values:
        norm = normalize_subcategory(value)
        if norm and norm not in normalized:
            normalized.append(norm)
    return normalized


def format_subcategory_cell(subcategories: Sequence[str]) -> str:
    normalized = normalize_subcategories(subcategories)
    if len(normalized) < 1 or len(normalized) > 3:
        raise ValueError("Subcategory must contain 1 to 3 tags.")
    return ";".join(normalized)


def validate_prediction(
    prediction: Prediction, min_confidence: float
) -> Tuple[bool, str]:
    if prediction.category not in TAXONOMY:
        return False, "invalid_category"
    if len(prediction.subcategories) < 1 or len(prediction.subcategories) > 3:
        return False, "invalid_subcategory_count"
    allowed_tags = set(TAXONOMY[prediction.category])
    for subcategory in prediction.subcategories:
        if subcategory not in allowed_tags:
            return False, f"invalid_subcategory:{subcategory}"
    if not (0.0 <= prediction.confidence <= 1.0):
        return False, "invalid_confidence_range"
    if prediction.confidence < min_confidence:
        return False, "low_confidence"
    return True, ""


def repair_category_subcategory_link(prediction: Prediction) -> Prediction:
    """If tags clearly belong to a different single category, repair category."""
    inferred_categories = {
        SUBCATEGORY_TO_CATEGORY.get(tag) for tag in prediction.subcategories if tag
    }
    inferred_categories.discard(None)
    if len(inferred_categories) != 1:
        return prediction
    inferred = next(iter(inferred_categories))
    if inferred == prediction.category:
        return prediction
    return Prediction(
        row_id=prediction.row_id,
        category=inferred,
        subcategories=prediction.subcategories,
        confidence=prediction.confidence,
    )


def chunked(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def read_csv_rows(input_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.startswith("."):
        stem = stem[1:]
    if stem.endswith("_original"):
        stem = stem[: -len("_original")] + "_categorized"
    else:
        stem = stem + ".categorized"
    return input_path.with_name(f"{stem}{input_path.suffix}")


def default_checkpoint_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.startswith("."):
        stem = stem[1:]
    if stem.endswith("_original"):
        stem = stem[: -len("_original")] + "_categorizer"
    else:
        stem = stem + "_categorizer"
    return Path(".cache") / f"{stem}.checkpoint.json"


def batch_schema() -> Dict[str, Any]:
    return {
        "name": "hsk_classification_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "row_id",
                            "category",
                            "subcategories",
                            "confidence",
                        ],
                        "properties": {
                            "row_id": {"type": "integer"},
                            "category": {
                                "type": "string",
                                "enum": list(TAXONOMY.keys()),
                            },
                            "subcategories": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "items": {
                                    "type": "string",
                                    "enum": sorted(
                                        {
                                            tag
                                            for tags in TAXONOMY.values()
                                            for tag in tags
                                        }
                                    ),
                                },
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                    },
                }
            },
        },
    }


def taxonomy_markdown() -> str:
    lines: List[str] = []
    for category, tags in TAXONOMY.items():
        lines.append(f"- {category}: {', '.join(tags)}")
    return "\n".join(lines)


def build_messages(
    rows_payload: Sequence[Dict[str, Any]],
    strict_retry: bool,
    retry_hints: Optional[Dict[int, str]],
) -> List[Dict[str, str]]:
    system_lines = [
        "You are a Mandarin HSK curriculum classifier.",
        "Classify each row into exactly one category and 1-3 subcategories.",
        "Use meaning from Chinese, pinyin, and English gloss.",
        "For discourse and grammar words, use Grammar & Function Words and the best matching grammar tags.",
        "Never invent categories or subcategories outside the taxonomy.",
        "Return all rows in the input once each.",
        "",
        "Taxonomy:",
        taxonomy_markdown(),
    ]
    if strict_retry:
        system_lines.extend(
            [
                "",
                "Retry mode:",
                "Previous answers failed validation or confidence threshold.",
                "Use only exact taxonomy tags and choose the closest valid labels.",
            ]
        )

    user_payload: Dict[str, Any] = {"rows": rows_payload}
    if retry_hints:
        user_payload["retry_hints"] = {
            str(row_id): reason for row_id, reason in retry_hints.items()
        }

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]


def parse_batch_response(payload: Dict[str, Any]) -> Dict[int, Prediction]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Model response missing 'items' array.")

    predictions: Dict[int, Prediction] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Model returned non-object batch item.")
        row_id = item.get("row_id")
        if not isinstance(row_id, int):
            raise ValueError("Model returned invalid row_id type.")
        if row_id in predictions:
            raise ValueError(f"Duplicate row_id in model response: {row_id}")

        category = normalize_category(item.get("category"))
        raw_subcategories = item.get("subcategories")
        if not isinstance(raw_subcategories, list):
            raise ValueError(f"Row {row_id} has invalid subcategories.")
        subcategories = normalize_subcategories(raw_subcategories)
        confidence_raw = item.get("confidence")
        confidence = float(confidence_raw)

        predictions[row_id] = Prediction(
            row_id=row_id,
            category=category,
            subcategories=subcategories,
            confidence=confidence,
        )
    return predictions


def call_openai_batch(
    client: OpenAI,
    model: str,
    rows_payload: Sequence[Dict[str, Any]],
    strict_retry: bool,
    retry_hints: Optional[Dict[int, str]],
    api_attempts: int,
    backoff_seconds: float,
) -> Dict[int, Prediction]:
    last_error: Optional[Exception] = None
    for attempt in range(1, api_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=build_messages(rows_payload, strict_retry, retry_hints),
                response_format={
                    "type": "json_schema",
                    "json_schema": batch_schema(),
                },
            )
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError(f"Model refusal: {message.refusal}")
            raw_content = message.content
            if not raw_content:
                raise RuntimeError("Model returned empty content.")
            parsed = json.loads(raw_content)
            return parse_batch_response(parsed)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == api_attempts:
                break
            sleep_for = backoff_seconds * attempt
            print(
                f"  API attempt {attempt}/{api_attempts} failed: {exc}. "
                f"Retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"OpenAI batch failed after {api_attempts} attempts: {last_error}")


def assess_batch(
    expected_ids: Sequence[int],
    batch_predictions: Dict[int, Prediction],
    min_confidence: float,
) -> Tuple[Dict[int, Prediction], Dict[int, str]]:
    accepted: Dict[int, Prediction] = {}
    rejected: Dict[int, str] = {}
    for row_id in expected_ids:
        prediction = batch_predictions.get(row_id)
        if prediction is None:
            rejected[row_id] = "missing_prediction"
            continue
        prediction = repair_category_subcategory_link(prediction)
        ok, reason = validate_prediction(prediction, min_confidence=min_confidence)
        if ok:
            accepted[row_id] = prediction
        else:
            rejected[row_id] = reason
    return accepted, rejected


def checkpoint_payload(
    predictions: Dict[int, Prediction],
    input_path: Path,
    input_sha256: str,
    model: str,
    min_confidence: float,
) -> Dict[str, Any]:
    return {
        "meta": {
            "input_path": str(input_path),
            "input_sha256": input_sha256,
            "model": model,
            "min_confidence": min_confidence,
            "updated_at_utc": now_iso(),
        },
        "predictions": {
            str(row_id): asdict(prediction)
            for row_id, prediction in sorted(predictions.items())
        },
    }


def save_checkpoint(
    checkpoint_path: Path,
    predictions: Dict[int, Prediction],
    input_path: Path,
    input_sha256: str,
    model: str,
    min_confidence: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        predictions=predictions,
        input_path=input_path,
        input_sha256=input_sha256,
        model=model,
        min_confidence=min_confidence,
    )
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    min_confidence: float,
) -> Dict[int, Prediction]:
    if not checkpoint_path.exists():
        return {}
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    raw_predictions = raw.get("predictions")
    if not isinstance(raw_predictions, dict):
        return {}
    loaded: Dict[int, Prediction] = {}
    for key, value in raw_predictions.items():
        try:
            row_id = int(key)
            if not isinstance(value, dict):
                continue
            prediction = Prediction(
                row_id=row_id,
                category=normalize_category(value.get("category")),
                subcategories=normalize_subcategories(value.get("subcategories", [])),
                confidence=float(value.get("confidence", 0)),
            )
            ok, _ = validate_prediction(prediction, min_confidence=min_confidence)
            if ok:
                loaded[row_id] = prediction
        except Exception:  # noqa: BLE001
            continue
    return loaded


def apply_predictions_to_rows(
    rows: Sequence[Dict[str, str]],
    predictions: Dict[int, Prediction],
) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for row_id, row in enumerate(rows):
        prediction = predictions.get(row_id)
        if prediction is None:
            raise ValueError(f"Missing prediction for row_id={row_id}")
        updated = dict(row)
        updated["Category"] = prediction.category
        updated["Subcategory"] = format_subcategory_cell(prediction.subcategories)
        enriched.append(updated)
    return enriched


def write_output_csv(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, str]],
) -> None:
    output_fields = list(fieldnames)
    if "Category" not in output_fields:
        output_fields.append("Category")
    if "Subcategory" not in output_fields:
        output_fields.append("Subcategory")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def make_row_payload(row_id: int, row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "row_id": row_id,
        "chinese": row.get("Chinese", ""),
        "pinyin": row.get("Pinyin", ""),
        "english": row.get("English", ""),
        "level": row.get("Level", ""),
    }


def classify_rows(
    client: OpenAI,
    rows: Sequence[Dict[str, str]],
    model: str,
    batch_size: int,
    retry_batch_size: int,
    max_retries: int,
    min_confidence: float,
    rescue_min_confidence: float,
    api_attempts: int,
    backoff_seconds: float,
    checkpoint_path: Path,
    input_path: Path,
    input_sha256: str,
) -> Dict[int, Prediction]:
    all_ids = list(range(len(rows)))
    all_id_set = set(all_ids)
    predictions = load_checkpoint(checkpoint_path, min_confidence=min_confidence)
    predictions = {row_id: pred for row_id, pred in predictions.items() if row_id in all_id_set}
    pending = [row_id for row_id in all_ids if row_id not in predictions]
    rejection_reasons: Dict[int, str] = {}

    if pending:
        print(f"Pass 1: classifying {len(pending)} rows in batches of {batch_size}...")
    for idx, batch_ids in enumerate(chunked(pending, batch_size), start=1):
        print(f"  Pass 1 batch {idx}: {len(batch_ids)} rows")
        rows_payload = [make_row_payload(row_id, rows[row_id]) for row_id in batch_ids]
        batch_predictions = call_openai_batch(
            client=client,
            model=model,
            rows_payload=rows_payload,
            strict_retry=False,
            retry_hints=None,
            api_attempts=api_attempts,
            backoff_seconds=backoff_seconds,
        )
        accepted, rejected = assess_batch(
            expected_ids=batch_ids,
            batch_predictions=batch_predictions,
            min_confidence=min_confidence,
        )
        predictions.update(accepted)
        rejection_reasons.update(rejected)
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            predictions=predictions,
            input_path=input_path,
            input_sha256=input_sha256,
            model=model,
            min_confidence=min_confidence,
        )

    pending = [row_id for row_id in all_ids if row_id not in predictions]

    for retry_round in range(1, max_retries + 1):
        if not pending:
            break
        print(
            f"Retry round {retry_round}/{max_retries}: "
            f"{len(pending)} rows in batches of {retry_batch_size}..."
        )
        for idx, batch_ids in enumerate(chunked(pending, retry_batch_size), start=1):
            hints = {row_id: rejection_reasons.get(row_id, "retry") for row_id in batch_ids}
            print(
                f"  Retry {retry_round} batch {idx}: {len(batch_ids)} rows "
                f"(hints: {', '.join(sorted(set(hints.values())))})"
            )
            rows_payload = [make_row_payload(row_id, rows[row_id]) for row_id in batch_ids]
            batch_predictions = call_openai_batch(
                client=client,
                model=model,
                rows_payload=rows_payload,
                strict_retry=True,
                retry_hints=hints,
                api_attempts=api_attempts,
                backoff_seconds=backoff_seconds,
            )
            accepted, rejected = assess_batch(
                expected_ids=batch_ids,
                batch_predictions=batch_predictions,
                min_confidence=min_confidence,
            )
            predictions.update(accepted)
            for row_id, reason in rejected.items():
                rejection_reasons[row_id] = reason
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                predictions=predictions,
                input_path=input_path,
                input_sha256=input_sha256,
                model=model,
                min_confidence=min_confidence,
            )
        pending = [row_id for row_id in all_ids if row_id not in predictions]

    if pending:
        print(
            f"Final rescue pass: {len(pending)} rows with min confidence "
            f"{rescue_min_confidence:.2f}..."
        )
        for idx, batch_ids in enumerate(chunked(pending, retry_batch_size), start=1):
            hints = {row_id: rejection_reasons.get(row_id, "final_rescue") for row_id in batch_ids}
            print(
                f"  Rescue batch {idx}: {len(batch_ids)} rows "
                f"(hints: {', '.join(sorted(set(hints.values())))})"
            )
            rows_payload = [make_row_payload(row_id, rows[row_id]) for row_id in batch_ids]
            batch_predictions = call_openai_batch(
                client=client,
                model=model,
                rows_payload=rows_payload,
                strict_retry=True,
                retry_hints=hints,
                api_attempts=api_attempts,
                backoff_seconds=backoff_seconds,
            )
            accepted, rejected = assess_batch(
                expected_ids=batch_ids,
                batch_predictions=batch_predictions,
                min_confidence=rescue_min_confidence,
            )
            predictions.update(accepted)
            for row_id, reason in rejected.items():
                rejection_reasons[row_id] = reason
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                predictions=predictions,
                input_path=input_path,
                input_sha256=input_sha256,
                model=model,
                min_confidence=min_confidence,
            )
        pending = [row_id for row_id in all_ids if row_id not in predictions]

    if pending:
        sample = ", ".join(str(row_id) for row_id in pending[:20])
        raise RuntimeError(
            f"Unresolved rows after retries: {len(pending)}. "
            f"Sample row_ids: {sample}. "
            "You can rerun; checkpoint is preserved."
        )

    return predictions


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            DEFAULT_OUTPUT_PATH.resolve()
            if input_path == DEFAULT_INPUT_PATH.resolve()
            else default_output_path(input_path).resolve()
        )
    )
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (
            DEFAULT_CHECKPOINT_PATH.resolve()
            if input_path == DEFAULT_INPUT_PATH.resolve()
            else default_checkpoint_path(input_path).resolve()
        )
    )

    if args.batch_size < 1 or args.retry_batch_size < 1:
        raise ValueError("batch-size and retry-batch-size must be >= 1")
    if args.max_retries < 0:
        raise ValueError("max-retries must be >= 0")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("min-confidence must be between 0 and 1")
    if not (0.0 <= args.rescue_min_confidence <= 1.0):
        raise ValueError("rescue-min-confidence must be between 0 and 1")
    if args.rescue_min_confidence > args.min_confidence:
        raise ValueError("rescue-min-confidence must be <= min-confidence")

    api_key = get_api_key(Path(args.env_file).expanduser().resolve())
    client = OpenAI(api_key=api_key)

    fieldnames, all_rows = read_csv_rows(input_path)
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    input_hash = sha256_file(input_path)

    print(f"Input: {input_path}")
    print(f"Rows to classify: {len(rows)}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")

    predictions = classify_rows(
        client=client,
        rows=rows,
        model=args.model,
        batch_size=args.batch_size,
        retry_batch_size=args.retry_batch_size,
        max_retries=args.max_retries,
        min_confidence=args.min_confidence,
        rescue_min_confidence=args.rescue_min_confidence,
        api_attempts=args.api_attempts,
        backoff_seconds=args.api_backoff_seconds,
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        input_sha256=input_hash,
    )

    enriched_rows = apply_predictions_to_rows(rows=rows, predictions=predictions)
    write_output_csv(output_path=output_path, fieldnames=fieldnames, rows=enriched_rows)
    print(f"Done. Wrote {len(enriched_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
