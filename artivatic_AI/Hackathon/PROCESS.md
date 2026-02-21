# Extraction Process

## Overview

The hospital bill extractor follows a five-stage pipeline that converts raw bill images and PDFs into structured JSON without any cloud API calls or per-request costs.

## Stage One: Preflight Validation

Before processing begins, the system checks that Tesseract OCR is installed and accessible, that all required Python libraries are present, and that the configured input directories exist. Any missing dependency causes the process to exit immediately with a clear error message rather than failing silently mid-run.

## Stage Two: OCR and Image Preprocessing

Each input file is passed through a preprocessing pipeline before OCR. The image is converted to grayscale, denoised using fast non-local means filtering, and then binarized using adaptive Gaussian thresholding. This approach handles uneven lighting and logo overlap artifacts common in scanned hospital documents. A deskew step measures the dominant text angle using minimum area rectangle fitting and rotates the image to correct any tilt beyond 0.3 degrees. PDFs are rendered at 300 DPI per page before the same preprocessing is applied. Small images under 1500 pixels on their longest side are upscaled by a factor of two or more before OCR to improve character recognition. Tesseract runs with page segmentation mode 6 and the LSTM engine in a configuration that preserves interword spacing.

## Stage Three: OCR Normalisation

Raw OCR output is corrected before any field parsing begins. A set of character substitution rules targets common Tesseract errors in Indian hospital documents, including digit-letter confusion inside alphanumeric codes, garbled punctuation from certain bill formats, and double-character OCR variants of medical terms. Applying these corrections at the text level means all downstream regex patterns work against cleaner input rather than needing to handle every OCR variant individually.

## Stage Four: Field Extraction

Each field is extracted by a dedicated function that tries a prioritized list of regex patterns in order and returns the first valid match. Fields with multiple common label variants are handled by including all known variants in the pattern list. Date fields use a four-level fallback strategy: explicit bill or invoice date labels, then print timestamps used in certain billing systems, then separately labelled admission and discharge fields, then a last-resort scan that assigns the first and last word-month dates found in the document. Amount fields handle the rupee symbol, comma separators, and embedded spaces before converting to float. Line items are extracted using five cascading patterns ranging from full four-column rows with SER service codes to logo-interrupted rows where the description and numeric data appear on separate lines. Pattern A validates that quantity multiplied by unit rate approximates the extended amount before accepting the column assignment, and swaps columns when the arithmetic points to a reversed layout.

## Stage Five: Validation and Output

After extraction, amount fields are cross-validated. If no gross total was found in the text, the system attempts to compute it by summing section totals or, as a final fallback, summing all individual line item amounts. The balance due is checked against gross total minus advance paid, and any discrepancy above five rupees is recorded as a warning inside the output JSON. Each processed file produces two outputs: a structured JSON file containing all extracted fields plus a metadata block with processing time and line item count, and a plain text file containing the raw OCR output for debugging. A summary report aggregating success and failure counts across the entire batch is written to the output directory on completion.

## Evaluation

The evaluation script loads extracted JSON files and compares them field by field against ground truth JSON files with matching filenames. Fields are classified as true positive, true negative, false positive, or false negative depending on whether the extracted and expected values are both present, both absent, or mismatched. Numeric fields use a one-rupee tolerance. String fields use case-insensitive whitespace-normalized comparison. Line items are matched using fuzzy description comparison on the first twenty characters after stripping service codes, combined with amount tolerance checking. The evaluator reports overall accuracy, precision, recall, F1 score, per-document breakdown, per-field breakdown, and an aggregate line-item accuracy percentage. Cost, response time, and static code quality metrics are included in the same dashboard output and saved to a JSON report file.
