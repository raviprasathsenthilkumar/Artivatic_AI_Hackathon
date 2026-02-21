#!/usr/bin/env python3
"""
Hospital Bill Extractor — Evaluation & Metrics Dashboard  v2
=============================================================
Computes and displays all hackathon evaluation criteria:

  ✓ Accuracy (TP, TN, FP, FN, Precision, Recall, F1)
  ✓ Cost Efficiency  (always $0.00 — free local OCR)
  ✓ Response Time    (per-file and aggregate)
  ✓ Code Quality     (static analysis)

KEY FIXES vs v1
───────────────
  FIX A — TN now counts toward accuracy numerator
    Previously: accuracy = (TP+TN)/total but TN was always 0 because
    the evaluator skipped fields where exp_val is None.
    Now:  fields where GT value is null/None correctly produce TN when
    the extractor also returns null (correct "not present" detection).

  FIX B — Boolean False evaluated correctly
    GT field "cashless": false — _is_empty(False) must return False
    so the field IS evaluated (was already correct, now explicitly tested).

  FIX C — Line-item accuracy section added
    New section compares extracted line items vs GT line items using
    fuzzy description matching + amount tolerance.
    Shows: items matched, missed, spurious, per-item accuracy %.

  FIX D — Smarter GT field skip logic
    Skip only when GT key is ABSENT from the GT dict entirely.
    If GT has "bill_number": null, that IS a valid expected value (null),
    not a signal to skip — it means we expect the extractor to return null.

USAGE:
    python evaluate_extractor.py
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
BASE             = Path(r"C:\Users\ravip\OneDrive\Desktop\artivatic_AI\Hackathon")
OUT_DIR          = BASE / "output"
GROUND_TRUTH     = BASE / "ground_truth"
EXTRACTOR_SCRIPT = Path(__file__).parent / "hospital_bill_extractor.py"

AMOUNT_TOL = 1.0   # rupees

# Fields evaluated — dot-notation for nested keys
EVAL_FIELDS = [
    "bill_type",
    "bill_number",
    "bill_date",
    "hospital.name",
    "hospital.gstin",
    "patient.name",
    "patient.age",
    "patient.gender",
    "admission.admission_date",
    "admission.discharge_date",
    "admission.los_days",
    "admission.ward",
    "clinical.attending_doctor",
    "charges_summary.gross_total",
    "charges_summary.balance_due",
    "charges_summary.discount",
    "charges_summary.advance_paid",
    "insurance.cashless",
    "insurance.policy_number",
]

NUMERIC_FIELDS = {
    "charges_summary.gross_total",
    "charges_summary.balance_due",
    "charges_summary.discount",
    "charges_summary.advance_paid",
    "admission.los_days",
    "patient.age",
}
# ══════════════════════════════════════════════════════════════


@dataclass
class FieldResult:
    """Outcome of evaluating one field in one document."""
    field:     str
    extracted: Any
    expected:  Any
    outcome:   str   # "TP" | "TN" | "FP" | "FN"


@dataclass
class DocumentMetrics:
    """Per-document evaluation results."""
    filename:      str
    processing_s:  float
    field_results: list[FieldResult] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return sum(1 for r in self.field_results if r.outcome == "TP")

    @property
    def tn(self) -> int:
        return sum(1 for r in self.field_results if r.outcome == "TN")

    @property
    def fp(self) -> int:
        return sum(1 for r in self.field_results if r.outcome == "FP")

    @property
    def fn(self) -> int:
        return sum(1 for r in self.field_results if r.outcome == "FN")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0


def _get(obj: dict, dotpath: str) -> Any:
    """Safely traverse nested dict with dot-notation key. Returns _MISSING sentinel if key absent."""
    keys = dotpath.split(".")
    for k in keys:
        if not isinstance(obj, dict):
            return _MISSING
        if k not in obj:
            return _MISSING
        obj = obj[k]
    return obj

_MISSING = object()   # sentinel for "key not in dict at all"


def _is_empty(v: Any) -> bool:
    """
    True if value represents 'not found' / missing.

    FIX A+B: Only None and empty string/list/dict count as empty.
    False (boolean) is NOT empty — it's a valid extracted value.
    The _MISSING sentinel is also not-empty from GT perspective
    (handled separately in evaluate_document).
    """
    if v is None:
        return True
    if v is _MISSING:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _values_match(extracted: Any, expected: Any, numeric: bool = False) -> bool:
    """
    Compare extracted vs expected with type-aware tolerance.

    Strings:  case-insensitive, whitespace-normalised
    Numbers:  within ±AMOUNT_TOL
    Booleans: direct equality (False == False → True)
    """
    if numeric and extracted is not None and expected is not None:
        try:
            return abs(float(extracted) - float(expected)) <= AMOUNT_TOL
        except (TypeError, ValueError):
            pass

    if isinstance(extracted, str) and isinstance(expected, str):
        return extracted.strip().lower() == expected.strip().lower()

    return extracted == expected


def _classify_outcome(extracted: Any, expected: Any, numeric: bool = False) -> str:
    """
    Classify field comparison into TP / TN / FP / FN.

    FIX A: Both null/empty → TN (counts toward accuracy).
    FIX D: expected=_MISSING means GT key absent → skip (handled upstream).

      TP — extracted matches expected (both present)
      TN — both extracted and expected are null/empty
      FP — extracted something wrong (expected null, got value;
            OR both present but value doesn't match)
      FN — expected a value, extracted nothing
    """
    ext_empty = _is_empty(extracted)
    exp_empty = _is_empty(expected)

    if exp_empty and ext_empty:
        return "TN"                                  # FIX A: correct "not present"
    if exp_empty and not ext_empty:
        return "FP"                                  # hallucinated a value
    if not exp_empty and ext_empty:
        return "FN"                                  # missed a real value
    return "TP" if _values_match(extracted, expected, numeric) else "FP"


def _fuzzy_desc_match(a: str, b: str) -> bool:
    """
    Fuzzy description match for line-item comparison.
    Strips SER codes, lowercases, compares first 20 chars.
    """
    def clean(s: str) -> str:
        s = re.sub(r"^\$?SER\w+\s+", "", s.strip(), flags=re.I)
        s = re.sub(r"[^a-z0-9 ]", "", s.lower())
        return s[:20].strip()
    return clean(a) == clean(b)


def _compare_line_items(
    extracted_items: list[dict],
    gt_items: list[dict],
) -> dict:
    """
    Compare extracted line items against ground-truth line items.

    Matching: fuzzy description (first 20 chars after stripping SER code)
    + amount within ±AMOUNT_TOL (checks both "amount" and "exc_amount").

    Returns dict with matched/missed/spurious counts and accuracy %.
    """
    matched   = 0
    missed    = 0
    spurious  = 0
    matched_gt = set()

    for ext in extracted_items:
        ext_desc = ext.get("description", "")
        # amount may be unit_rate or exc_amount depending on extractor convention
        ext_amts = {ext.get("amount"), ext.get("exc_amount"), ext.get("unit_rate")} - {None}

        found = False
        for j, gt in enumerate(gt_items):
            if j in matched_gt:
                continue
            gt_desc = gt.get("description", "")
            # GT stores both "amount" (unit price) and "exc_amount" (total)
            gt_amts = {gt.get("amount"), gt.get("exc_amount")} - {None}

            if _fuzzy_desc_match(ext_desc, gt_desc):
                # Check if any amount combination matches
                amt_match = any(
                    abs(ea - ga) <= max(AMOUNT_TOL, ga * 0.01)
                    for ea in ext_amts for ga in gt_amts
                    if ea is not None and ga is not None
                )
                if amt_match:
                    matched += 1
                    matched_gt.add(j)
                    found = True
                    break

        if not found:
            spurious += 1

    missed = len(gt_items) - len(matched_gt)

    total    = len(gt_items)
    accuracy = matched / total * 100 if total else 0.0

    return {
        "gt_count":      total,
        "extracted_count": len(extracted_items),
        "matched":       matched,
        "missed":        missed,
        "spurious":      spurious,
        "item_accuracy": round(accuracy, 1),
    }


def evaluate_document(
    extracted: dict,
    ground_truth: dict,
    filename: str,
    proc_time: float,
) -> tuple[DocumentMetrics, dict]:
    """
    Compare one extracted JSON against its ground truth.

    FIX D: A GT field is skipped only if the key is COMPLETELY ABSENT from
    the GT dict. If GT has "bill_number": null, that means "we expect null"
    and the field IS evaluated.
    """
    doc = DocumentMetrics(filename=filename, processing_s=proc_time)

    for fpath in EVAL_FIELDS:
        exp_val = _get(ground_truth, fpath)

        # FIX D: skip only when the key is absent from GT dict
        if exp_val is _MISSING:
            continue

        ext_val = _get(extracted, fpath)
        if ext_val is _MISSING:
            ext_val = None

        is_num  = fpath in NUMERIC_FIELDS
        outcome = _classify_outcome(ext_val, exp_val, numeric=is_num)
        doc.field_results.append(
            FieldResult(field=fpath, extracted=ext_val,
                        expected=exp_val, outcome=outcome)
        )

    # FIX C: Line-item comparison
    gt_items  = ground_truth.get("line_items", [])
    ext_items = extracted.get("line_items", [])
    item_stats = _compare_line_items(ext_items, gt_items) if gt_items else {}

    return doc, item_stats


def coverage_report(extracted_files: list[Path]) -> dict:
    """Field coverage when no ground truth is available."""
    totals: dict[str, int] = {f: 0 for f in EVAL_FIELDS}
    counts: dict[str, int] = {f: 0 for f in EVAL_FIELDS}

    for fp in extracted_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        for fpath in EVAL_FIELDS:
            totals[fpath] += 1
            v = _get(data, fpath)
            if not _is_empty(v) and v is not _MISSING:
                counts[fpath] += 1

    return {
        f: {
            "extracted_count": counts[f],
            "total_docs":      totals[f],
            "coverage_pct":    round(counts[f] / totals[f] * 100 if totals[f] else 0, 1),
        }
        for f in EVAL_FIELDS
    }


def _count_lines(fp: Path) -> dict:
    """Count code/comment/blank lines."""
    try:
        src   = fp.read_text(encoding="utf-8")
        lines = src.splitlines()
        code = blank = comment = 0
        for ln in lines:
            s = ln.strip()
            if not s:           blank += 1
            elif s.startswith("#"): comment += 1
            else:               code += 1
        return {
            "total_lines":   len(lines),
            "code_lines":    code,
            "comment_lines": comment,
            "blank_lines":   blank,
            "comment_ratio": round(comment / max(code, 1) * 100, 1),
        }
    except Exception:
        return {}


def _count_functions(fp: Path) -> dict:
    """Count functions/classes/docstrings via AST."""
    try:
        tree    = ast.parse(fp.read_text(encoding="utf-8"))
        funcs   = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        documented = sum(1 for f in funcs if ast.get_docstring(f))
        return {
            "functions":         len(funcs),
            "classes":           len(classes),
            "documented_funcs":  documented,
            "documentation_pct": round(documented / max(len(funcs), 1) * 100, 1),
        }
    except Exception:
        return {}


def analyze_code_quality(script: Path) -> dict:
    """Static code quality metrics — line counts, function coverage, maintainability score."""
    if not script.exists():
        return {"error": f"Script not found: {script}"}
    lines = _count_lines(script)
    funcs = _count_functions(script)
    avg_func_len = lines.get("code_lines", 0) / max(funcs.get("functions", 1), 1)
    score = 100.0
    if lines.get("comment_ratio", 0) < 10:        score -= 10
    if funcs.get("documentation_pct", 0) < 50:    score -= 15
    if avg_func_len > 50:                          score -= 10
    if lines.get("total_lines", 0) > 1000:        score -= 5
    return {
        "file":             script.name,
        "line_metrics":     lines,
        "function_metrics": funcs,
        "avg_lines_per_fn": round(avg_func_len, 1),
        "maintainability":  round(min(score, 100), 1),
    }


# ─── Report rendering ─────────────────────────────────────────

W = 68

def _hr(c: str = "─") -> str:   return c * W
def _center(t: str) -> str:     return t.center(W)
def _row(label: str, value: Any, width: int = 40) -> str:
    label, value = str(label), str(value)
    return f"  {label}{'.' * max(1, width - len(label))} {value}"
def _bar(pct: float, width: int = 30) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_full_report(
    doc_metrics:  list[DocumentMetrics],
    item_stats_list: list[dict],
    coverage:     dict | None,
    timing:       dict,
    code_quality: dict,
    has_gt:       bool,
) -> dict:
    """Render the full evaluation dashboard and return the report dict."""

    print("\n" + "═" * W)
    print(_center("  HOSPITAL BILL EXTRACTOR — EVALUATION REPORT  "))
    print("═" * W)

    # ── SECTION 1: Accuracy ──────────────────────────────────
    print(f"\n{'▌ ACCURACY METRICS':^{W}}")
    print(_hr())

    if has_gt and doc_metrics:
        total_tp = sum(d.tp for d in doc_metrics)
        total_tn = sum(d.tn for d in doc_metrics)
        total_fp = sum(d.fp for d in doc_metrics)
        total_fn = sum(d.fn for d in doc_metrics)
        total    = total_tp + total_tn + total_fp + total_fn

        overall_acc = (total_tp + total_tn) / total if total else 0
        precision   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        recall      = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        print(_row("True  Positives  (TP) — Correct extractions", total_tp))
        print(_row("True  Negatives  (TN) — Correct 'not present'", total_tn))
        print(_row("False Positives  (FP) — Wrong value extracted", total_fp))
        print(_row("False Negatives  (FN) — Missed real values", total_fn))
        print(_hr("·"))
        print(_row("Overall Accuracy", f"{overall_acc*100:.2f}%"))
        print(_row("Precision", f"{precision*100:.2f}%"))
        print(_row("Recall   (Sensitivity)", f"{recall*100:.2f}%"))
        print(_row("F1 Score", f"{f1*100:.2f}%"))
        print(f"\n  Accuracy  {_bar(overall_acc*100)}  {overall_acc*100:.1f}%")

        threshold_met = overall_acc >= 0.95
        status = "✅ PASS — ≥95% threshold met" if threshold_met else "❌ FAIL — below 95% threshold"
        print(f"\n  Qualification Status: {status}")

        # Per-document breakdown
        print(f"\n  {'─ Per-Document Breakdown ':─<{W-2}}")
        print(f"  {'File':<28} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc':>7} {'F1':>7}")
        print(f"  {'-'*62}")
        for dm in doc_metrics:
            print(
                f"  {dm.filename[:27]:<28}"
                f" {dm.tp:>4} {dm.tn:>4} {dm.fp:>4} {dm.fn:>4}"
                f" {dm.accuracy*100:>6.1f}% {dm.f1*100:>6.1f}%"
            )

        # Per-field breakdown
        print(f"\n  {'─ Per-Field Accuracy ':─<{W-2}}")
        field_stats: dict[str, dict] = {}
        for dm in doc_metrics:
            for fr in dm.field_results:
                s = field_stats.setdefault(fr.field, {"TP":0,"TN":0,"FP":0,"FN":0})
                s[fr.outcome] += 1

        print(f"  {'Field':<42} {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} {'Acc':>7}")
        print(f"  {'-'*63}")
        for fname, s in sorted(field_stats.items()):
            t   = sum(s.values())
            acc = (s["TP"] + s["TN"]) / t * 100 if t else 0
            flag = " ⚠" if acc < 80 else ""
            print(f"  {fname:<42} {s['TP']:>3} {s['TN']:>3} {s['FP']:>3} {s['FN']:>3} {acc:>6.0f}%{flag}")

        # FIX C: Line-item accuracy
        if item_stats_list:
            print(f"\n  {'─ Line-Item Accuracy ':─<{W-2}}")
            agg = {k: sum(d.get(k, 0) for d in item_stats_list)
                   for k in ["gt_count","extracted_count","matched","missed","spurious"]}
            item_acc = agg["matched"] / agg["gt_count"] * 100 if agg["gt_count"] else 0
            print(_row("GT line items", agg["gt_count"]))
            print(_row("Extracted line items", agg["extracted_count"]))
            print(_row("Matched (desc+amount)", agg["matched"]))
            print(_row("Missed (FN)", agg["missed"]))
            print(_row("Spurious (FP)", agg["spurious"]))
            print(f"\n  Item Accuracy  {_bar(item_acc)}  {item_acc:.1f}%")

    else:
        print("  ℹ  No ground truth found — showing FIELD COVERAGE instead.")
        if coverage:
            print(f"\n  {'Field':<42} {'Coverage':>10}  Bar")
            print(f"  {'-'*65}")
            for fname, info in coverage.items():
                pct  = info["coverage_pct"]
                flag = " ⚠" if pct < 60 else ""
                print(f"  {fname:<42} {pct:>8.1f}%  {_bar(pct, 18)}{flag}")
            avg_cov = statistics.mean(v["coverage_pct"] for v in coverage.values())
            print(f"\n  Average field coverage: {avg_cov:.1f}%")

    # ── SECTION 2: Response Time ──────────────────────────────
    print(f"\n\n{'▌ RESPONSE TIME METRICS':^{W}}")
    print(_hr())
    print(_row("Files processed", timing["total_files"]))
    print(_row("Total wall time", f"{timing['total_s']:.2f}s"))
    print(_row("Average per file", f"{timing['avg_s']:.2f}s"))
    print(_row("Fastest file", f"{timing['min_s']:.2f}s"))
    print(_row("Slowest file", f"{timing['max_s']:.2f}s"))
    if "p50_s" in timing:
        print(_row("Median (P50)", f"{timing['p50_s']:.2f}s"))
        print(_row("P90 response time", f"{timing['p90_s']:.2f}s"))
    avg = timing["avg_s"]
    speed = ("🚀 Excellent (< 5s/doc)"  if avg < 5 else
             "✅ Good (< 15s/doc)"      if avg < 15 else
             "⚠  Acceptable (< 30s/doc)" if avg < 30 else
             "❌ Slow (> 30s/doc)")
    print(f"\n  Speed Rating: {speed}")

    # ── SECTION 3: Cost ────────────────────────────────────────
    print(f"\n\n{'▌ COST EFFICIENCY':^{W}}")
    print(_hr())
    print(_row("OCR engine", "Tesseract (local, free)"))
    print(_row("API calls", "0"))
    print(_row("Cost per document", "$0.00"))
    print(_row("Total cost", "$0.00"))
    print(_row("Estimated OpenAI GPT-4V equivalent", f"~${timing['total_files'] * 0.03:.2f}"))
    print(_row("Savings vs cloud OCR", "100%"))
    print("\n  Cost Rating: 🏆 Maximum (free local processing)")

    # ── SECTION 4: Code Quality ────────────────────────────────
    print(f"\n\n{'▌ CODE QUALITY METRICS':^{W}}")
    print(_hr())
    if "error" not in code_quality:
        lm = code_quality.get("line_metrics", {})
        fm = code_quality.get("function_metrics", {})
        print(_row("Script", code_quality.get("file", "N/A")))
        print(_row("Total lines", lm.get("total_lines", "N/A")))
        print(_row("Code lines", lm.get("code_lines", "N/A")))
        print(_row("Comment lines", lm.get("comment_lines", "N/A")))
        print(_row("Comment ratio", f"{lm.get('comment_ratio',0):.1f}%"))
        print(_hr("·"))
        print(_row("Functions defined", fm.get("functions", "N/A")))
        print(_row("Classes defined", fm.get("classes", "N/A")))
        print(_row("Functions with docstrings",
                   f"{fm.get('documented_funcs','N/A')} / {fm.get('functions','?')} "
                   f"({fm.get('documentation_pct',0):.0f}%)"))
        print(_row("Average lines per function", f"{code_quality.get('avg_lines_per_fn','N/A')}"))
        maint = code_quality.get("maintainability", 0)
        print(f"\n  Maintainability  {_bar(maint)}  {maint:.0f}/100")
        grade = ("A (Excellent)" if maint >= 85 else "B (Good)" if maint >= 70 else
                 "C (Fair)"     if maint >= 55 else "D (Needs work)")
        print(f"  Grade: {grade}")
    else:
        print(f"  ⚠  {code_quality['error']}")

    # ── SECTION 5: Summary ────────────────────────────────────
    print(f"\n\n{'▌ SCORECARD SUMMARY':^{W}}")
    print(_hr("═"))

    scores: dict[str, float] = {}
    if has_gt and doc_metrics:
        scores["Accuracy"] = overall_acc * 100
        if item_stats_list:
            scores["Line-Item Accuracy"] = item_acc
    else:
        avg_cov = statistics.mean(v["coverage_pct"] for v in coverage.values()) if coverage else 0
        scores["Field Coverage (proxy)"] = avg_cov

    scores["Cost Efficiency"] = 100.0
    scores["Response Time"]   = min(100, max(0, 100 - (timing["avg_s"] / 30) * 50))
    scores["Code Quality"]    = code_quality.get("maintainability", 70)

    for metric, score in scores.items():
        print(f"  {metric:<25}  {_bar(score, 28)}  {score:.1f}/100")
    overall_score = statistics.mean(scores.values())
    print(_hr("·"))
    print(f"  {'OVERALL SCORE':<25}  {_bar(overall_score, 28)}  {overall_score:.1f}/100")
    print("═" * W + "\n")

    # Build report dict
    report: dict[str, Any] = {
        "generated_at":  __import__("datetime").datetime.now().isoformat(),
        "timing":        timing,
        "cost":          {"total_usd": 0.0, "per_doc": 0.0, "engine": "tesseract_local"},
        "code_quality":  code_quality,
        "scores":        {k: round(v, 2) for k, v in scores.items()},
        "overall_score": round(overall_score, 2),
    }

    if has_gt and doc_metrics:
        report["accuracy"] = {
            "TP": total_tp, "TN": total_tn, "FP": total_fp, "FN": total_fn,
            "overall_pct":   round(overall_acc * 100, 2),
            "precision_pct": round(precision   * 100, 2),
            "recall_pct":    round(recall      * 100, 2),
            "f1_pct":        round(f1          * 100, 2),
            "threshold_met": threshold_met,
        }
        report["per_document"] = [
            {"file": dm.filename, "time_s": dm.processing_s,
             "TP": dm.tp, "TN": dm.tn, "FP": dm.fp, "FN": dm.fn,
             "accuracy": round(dm.accuracy*100,2), "f1": round(dm.f1*100,2)}
            for dm in doc_metrics
        ]
        if item_stats_list:
            report["line_item_accuracy"] = agg
    else:
        report["field_coverage"] = coverage

    return report


def main() -> None:
    """Main entry point — load files, run evaluation, print and save report."""
    print("=" * W)
    print(_center("  EVALUATION ENGINE STARTING  "))
    print("=" * W)

    extracted_files = sorted([
        f for f in OUT_DIR.glob("*.json") if not f.name.startswith("_")
    ])
    if not extracted_files:
        print(f"\n  ✗  No extracted JSON files found in {OUT_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(extracted_files)} extracted file(s) in {OUT_DIR}")

    times: list[float] = []
    for fp in extracted_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            t    = data.get("_meta", {}).get("processing_time_s")
            if t is not None:
                times.append(float(t))
        except Exception:
            pass
    if not times:
        times = [0.0]

    timing: dict[str, Any] = {
        "total_files": len(extracted_files),
        "total_s":     round(sum(times), 2),
        "avg_s":       round(statistics.mean(times), 2),
        "min_s":       round(min(times), 2),
        "max_s":       round(max(times), 2),
    }
    if len(times) >= 2:
        st = sorted(times)
        timing["p50_s"] = round(statistics.median(st), 2)
        timing["p90_s"] = round(st[math.ceil(len(st) * 0.9) - 1], 2)

    gt_files = list(GROUND_TRUTH.glob("*.json")) if GROUND_TRUTH.exists() else []
    has_gt   = len(gt_files) > 0

    doc_metrics:     list[DocumentMetrics] = []
    item_stats_list: list[dict]            = []
    coverage:        dict | None           = None

    if has_gt:
        print(f"  Found {len(gt_files)} ground-truth file(s) — running full accuracy evaluation.\n")
        for gt_fp in gt_files:
            ext_fp = OUT_DIR / gt_fp.name
            if not ext_fp.exists():
                print(f"  ⚠  No extracted file for GT: {gt_fp.name} — skipping")
                continue
            try:
                extracted  = json.loads(ext_fp.read_text(encoding="utf-8"))
                gt         = json.loads(gt_fp.read_text(encoding="utf-8"))
                proc_time  = extracted.get("_meta", {}).get("processing_time_s", 0)
                dm, istats = evaluate_document(extracted, gt, gt_fp.name, proc_time)
                doc_metrics.append(dm)
                if istats:
                    item_stats_list.append(istats)
            except Exception as exc:
                print(f"  ✗  Error evaluating {gt_fp.name}: {exc}")
    else:
        print("  ℹ  No ground truth directory found — computing field coverage.\n")
        coverage = coverage_report(extracted_files)

    cq     = analyze_code_quality(EXTRACTOR_SCRIPT)
    report = print_full_report(doc_metrics, item_stats_list, coverage, timing, cq, has_gt)

    report_path = OUT_DIR / "_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Full report saved → {report_path}\n")


if __name__ == "__main__":
    main()