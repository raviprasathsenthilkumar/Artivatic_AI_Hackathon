# Hospital Bill Extractor

An automated OCR-based pipeline for extracting structured data from Indian hospital bills. Built for the Artivatic AI Hackathon with a target of 95 percent or higher field-level accuracy at zero API cost.

## Project Overview

This project processes hospital bill images and PDFs using local Tesseract OCR and a custom regex extraction engine. It produces structured JSON output covering patient details, admission data, charge summaries, insurance fields, and itemized line charges. A companion evaluation script scores all outputs against ground truth using standard classification metrics.

## Repository Structure

```
hospital-bill-extractor/
├── hospital_bill_extractor.py   # Main extraction pipeline (v6)
├── evaluate_extractor.py        # Evaluation and metrics dashboard (v2)
├── Sample data/                 # Input images (JPG, PNG, TIFF)
├── Testing data/                # Input PDFs
├── output/                      # Extracted JSON and OCR text files
└── ground_truth/                # Ground truth JSON files for evaluation
```

## Requirements

Python 3.10 or higher is required. Install dependencies with:

```
pip install pytesseract pillow pymupdf opencv-python
```

Tesseract OCR must be installed separately. Download from the official Tesseract repository and update the TESSERACT_CMD path in the extractor script to match your installation.

## Configuration

Edit the following constants at the top of hospital_bill_extractor.py before running:

- BASE: root directory for all data folders
- IMG_DIR: folder containing image inputs
- PDF_DIR: folder containing PDF inputs
- OUT_DIR: folder where JSON outputs are written
- TESSERACT_CMD: full path to the Tesseract executable

## Usage

Run the extractor to process all files in the configured input directories:

```
python hospital_bill_extractor.py
```

Run the evaluator after extraction to score results against ground truth:

```
python evaluate_extractor.py
```

If a ground truth directory is present, the evaluator reports full accuracy metrics. Without ground truth, it reports field coverage percentages.

## Extracted Fields

The extractor captures the following structured fields from each document:

- Bill metadata: bill type, bill number, bill date
- Hospital: name, address, phone, email, GSTIN
- Patient: name, age, gender, date of birth, UHID
- Admission: admission date, discharge date, ward, bed number, length of stay
- Clinical: attending doctor, department, diagnoses
- Charges summary: gross total, balance due, advance paid, discount, GST components
- Insurance: TPA name, policy number, cashless flag
- Line items: description, category, quantity, unit rate, extended amount
- Payments: mode and amount for each payment entry

## Evaluation Metrics

The evaluation dashboard reports accuracy as (TP + TN) divided by total evaluated fields. It also reports precision, recall, F1, per-document breakdown, per-field breakdown, and line-item matching accuracy using fuzzy description comparison. Response time is measured per file and in aggregate. Code quality is assessed via static analysis of comment ratio, docstring coverage, and average function length.

## Performance Targets

- Accuracy: 95 percent or higher overall field accuracy
- Cost: zero dollars per document using local Tesseract
- Speed: under 5 seconds per document on standard hardware
- Code quality: Grade A maintainability score

## Key Technical Features

The extraction pipeline applies OCR character correction before regex parsing, handling common Tesseract errors such as digit-letter confusion and garbled punctuation. Date extraction uses a four-level fallback strategy including DOTALL multiline matching to handle OCR line splits. Line-item parsing uses five cascading patterns with quantity-rate-amount arithmetic validation to resolve column order ambiguity. Amount validation cross-checks balance due against gross total minus advance paid and flags discrepancies.

## License

This project was developed for the Artivatic AI Hackathon. Refer to the competition terms for usage rights.
