"""
Analytical Metrics and Statistical Aggregation Engine for SaveOnCare.

Calculates executive KPIs, benchmark coverage rates (item and value-based),
potential discrepancy rates and values, category breakdowns, and synthetic validation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ExecutiveMetrics:
    total_bills: int
    total_line_items: int
    total_billed_amount_inr: float
    comparable_line_items: int
    incomparable_line_items: int
    item_benchmark_coverage_pct: float
    value_benchmark_coverage_pct: float
    comparable_billed_amount_inr: float
    potential_discrepancy_count: int
    potential_discrepancy_rate_pct: float
    potential_discrepancy_value_inr: float
    discrepancy_share_of_billed_pct: float
    average_variance_pct_comparable: float  # item-weighted average
    amount_weighted_variance_pct: float     # amount-weighted average
    discrepancy_threshold_pct: float


@dataclass
class TestValidationMetrics:
    has_test_labels: bool
    total_labeled_items: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    interpretation_notes: str


def calculate_executive_metrics(
    processed_df: pd.DataFrame,
    threshold_pct: float = 20.0,
) -> ExecutiveMetrics:
    """
    Computes top-level executive metrics from processed billing dataframe.
    """
    total_items = len(processed_df)
    if total_items == 0:
        return ExecutiveMetrics(
            total_bills=0,
            total_line_items=0,
            total_billed_amount_inr=0.0,
            comparable_line_items=0,
            incomparable_line_items=0,
            item_benchmark_coverage_pct=0.0,
            value_benchmark_coverage_pct=0.0,
            comparable_billed_amount_inr=0.0,
            potential_discrepancy_count=0,
            potential_discrepancy_rate_pct=0.0,
            potential_discrepancy_value_inr=0.0,
            discrepancy_share_of_billed_pct=0.0,
            average_variance_pct_comparable=0.0,
            amount_weighted_variance_pct=0.0,
            discrepancy_threshold_pct=threshold_pct,
        )

    total_bills = int(processed_df["bill_id"].nunique()) if "bill_id" in processed_df.columns else 0

    # Total billed amount includes all rows with valid positive billed amount
    valid_billed_mask = processed_df["billed_line_amount"].notnull() & (processed_df["billed_line_amount"] > 0)
    total_billed_amount = float(processed_df.loc[valid_billed_mask, "billed_line_amount"].sum())

    # Comparable subsets
    comp_mask = processed_df["is_comparable"] == True
    comparable_count = int(comp_mask.sum())
    incomparable_count = total_items - comparable_count

    comparable_billed_amount = float(
        processed_df.loc[comp_mask & valid_billed_mask, "billed_line_amount"].sum()
    )

    # Benchmark coverage:
    # 1. Item coverage: comparable count / total count
    item_coverage_pct = (comparable_count / total_items) * 100.0 if total_items > 0 else 0.0

    # 2. Value coverage: comparable billed amount / total billed amount
    value_coverage_pct = (
        (comparable_billed_amount / total_billed_amount) * 100.0 if total_billed_amount > 0 else 0.0
    )

    # Potential discrepancy metrics on comparable items
    discrepancy_mask = comp_mask & (processed_df["is_potential_discrepancy"] == True)
    discrepancy_count = int(discrepancy_mask.sum())

    discrepancy_rate_pct = (
        (discrepancy_count / comparable_count) * 100.0 if comparable_count > 0 else 0.0
    )

    # Potential discrepancy value: sum of positive variance amounts for flagged items
    discrepancy_value = float(
        processed_df.loc[discrepancy_mask, "potential_discrepancy_value_inr"].sum()
    )

    discrepancy_share_pct = (
        (discrepancy_value / total_billed_amount) * 100.0 if total_billed_amount > 0 else 0.0
    )

    # Variance averages:
    # A. Line-item unweighted average
    avg_var_item = (
        float(processed_df.loc[comp_mask, "variance_pct"].mean()) if comparable_count > 0 else 0.0
    )

    # B. Amount-weighted average variance:
    # (Total Comparable Billed - Total Comparable Reference) / Total Comparable Reference * 100
    comp_ref_total = float(processed_df.loc[comp_mask, "reference_line_amount"].sum())
    if comp_ref_total > 0:
        amount_weighted_var = ((comparable_billed_amount - comp_ref_total) / comp_ref_total) * 100.0
    else:
        amount_weighted_var = 0.0

    return ExecutiveMetrics(
        total_bills=total_bills,
        total_line_items=total_items,
        total_billed_amount_inr=round(total_billed_amount, 2),
        comparable_line_items=comparable_count,
        incomparable_line_items=incomparable_count,
        item_benchmark_coverage_pct=round(item_coverage_pct, 1),
        value_benchmark_coverage_pct=round(value_coverage_pct, 1),
        comparable_billed_amount_inr=round(comparable_billed_amount, 2),
        potential_discrepancy_count=discrepancy_count,
        potential_discrepancy_rate_pct=round(discrepancy_rate_pct, 1),
        potential_discrepancy_value_inr=round(discrepancy_value, 2),
        discrepancy_share_of_billed_pct=round(discrepancy_share_pct, 1),
        average_variance_pct_comparable=round(avg_var_item, 1),
        amount_weighted_variance_pct=round(amount_weighted_var, 1),
        discrepancy_threshold_pct=threshold_pct,
    )


def calculate_category_summary(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes granular category-level breakdown:
    - Total line items
    - Total billed amount
    - Comparable items & item coverage %
    - Value coverage %
    - Potential discrepancy items & rate %
    - Potential discrepancy value (INR)
    - Item-weighted average variance %
    """
    categories = sorted(processed_df["category"].unique())
    summary_rows = []

    for cat in categories:
        sub = processed_df[processed_df["category"] == cat]
        total_items = len(sub)
        total_billed = float(sub["billed_line_amount"].sum())

        comp_sub = sub[sub["is_comparable"] == True]
        comp_count = len(comp_sub)
        comp_billed = float(comp_sub["billed_line_amount"].sum())

        item_cov = (comp_count / total_items) * 100 if total_items > 0 else 0.0
        val_cov = (comp_billed / total_billed) * 100 if total_billed > 0 else 0.0

        disc_sub = comp_sub[comp_sub["is_potential_discrepancy"] == True]
        disc_count = len(disc_sub)
        disc_rate = (disc_count / comp_count) * 100 if comp_count > 0 else 0.0
        disc_val = float(disc_sub["potential_discrepancy_value_inr"].sum())

        avg_var = float(comp_sub["variance_pct"].mean()) if comp_count > 0 else 0.0

        summary_rows.append(
            {
                "Category": cat.capitalize(),
                "Total Items": total_items,
                "Total Billed (₹)": round(total_billed, 2),
                "Comparable Items": comp_count,
                "Item Coverage (%)": round(item_cov, 1),
                "Value Coverage (%)": round(val_cov, 1),
                "Discrepancy Items": disc_count,
                "Discrepancy Rate (%)": round(disc_rate, 1),
                "Potential Discrepancy Value (₹)": round(disc_val, 2),
                "Avg Variance (%)": round(avg_var, 1),
            }
        )

    return pd.DataFrame(summary_rows)


def evaluate_test_dataset_performance(
    processed_df: pd.DataFrame,
) -> TestValidationMetrics:
    """
    Evaluates the discrepancy detection rule against ground truth injected test labels.
    Calculates precision, recall, F1-score, and confusion matrix.
    """
    if "injected_discrepancy" not in processed_df.columns:
        return TestValidationMetrics(
            has_test_labels=False,
            total_labeled_items=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            interpretation_notes="No ground truth injected_discrepancy labels found in dataset.",
        )

    # Evaluate exclusively on comparable items where the rule was actually applied
    eval_df = processed_df[processed_df["is_comparable"] == True].copy()
    total_labeled = len(eval_df)

    if total_labeled == 0:
        return TestValidationMetrics(
            has_test_labels=True,
            total_labeled_items=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            interpretation_notes="No comparable items available for validation evaluation.",
        )

    y_true = eval_df["injected_discrepancy"].astype(bool)
    y_pred = eval_df["is_potential_discrepancy"].astype(bool)

    tp = int(((y_pred == True) & (y_true == True)).sum())
    fp = int(((y_pred == True) & (y_true == False)).sum())
    tn = int(((y_pred == False) & (y_true == False)).sum())
    fn = int(((y_pred == False) & (y_true == True)).sum())

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100.0 if (tp + tn + fp + fn) > 0 else 0.0

    notes = (
        "Validation evaluated on synthetically injected test anomalies. "
        "A true positive corresponds to an injected discrepancy successfully flagged by the threshold rule. "
        "False positives represent standard items whose legitimate hospital markup exceeded the chosen threshold. "
        "CRITICAL CAVEAT: High precision/recall on this synthetic benchmark demonstrates internal consistency "
        "of the threshold rule but does NOT prove real-world accuracy on unstructured hospital bills without "
        "independent clinical validation."
    )

    return TestValidationMetrics(
        has_test_labels=True,
        total_labeled_items=total_labeled,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 1),
        recall=round(recall, 1),
        f1_score=round(f1, 1),
        accuracy=round(accuracy, 1),
        interpretation_notes=notes,
    )
