# HSK 1-2-3 Vocabulary Categorization

OpenAI-powered classification pipeline for HSK 1-2-3 vocabulary entries.

It assigns:
- one `Category` label
- one to three `Subcategory` tags (semicolon-delimited)

The taxonomy is fixed to 10 student-focused categories, including grammar-oriented subcategories such as `contrast`, `addition`, and `sequence`.

## Repository layout

```text
.
├── data/
│   ├── hsk1-2-3_vocab_categorized.csv
│   └── .hsk1-2-3_vocab_original.csv
├── hsk_csv_categorizer.py
├── tests/
│   └── test_hsk_csv_categorizer.py
└── README.md
```

Notes:
- `data/.hsk1-2-3_vocab_original.csv` is kept as a hidden backup source file.
- `data/hsk1-2-3_vocab_categorized.csv` is the final enriched CSV intended for use.

## Requirements

- Python 3.10+
- `openai` Python package
- OpenAI API key in one of:
  - `OPENAI_API_KEY`
  - `OPENAIAPI_KEY`
  - `.env` file in repo root

Install dependency:

```bash
python3 -m pip install openai
```

## Usage

Smoke test (cheap, small sample):

```bash
python3 hsk_csv_categorizer.py --limit 20 --batch-size 10 --retry-batch-size 5 --max-retries 1
```

Full run:

```bash
python3 hsk_csv_categorizer.py
```

Useful flags:

- `--model gpt-4.1-mini` (default)
- `--min-confidence 0.72`
- `--rescue-min-confidence 0.55`
- `--input <path/to/input.csv>`
- `--output <path/to/output.csv>`
- `--checkpoint <path/to/checkpoint.json>`

## Output format

The output CSV preserves all original columns and ensures:
- `Category` is always populated
- `Subcategory` is always populated
- `Subcategory` contains 1-3 tags, joined with `;`
- each subcategory is validated against the selected category

## Quality and reliability features

- strict JSON schema response parsing
- category/subcategory consistency validation
- automatic retry for invalid or low-confidence rows
- final rescue pass for stubborn rows
- checkpoint/resume support to avoid paying twice after interruptions

## Tests

Run unit tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
