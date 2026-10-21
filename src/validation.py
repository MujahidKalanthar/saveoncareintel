"""
Data Validation and Quality Assurance Pipeline for SaveOnCare.

Performs deterministic data validation, missing value profiling, duplicate detection,
quantity/unit audits, and arithmetic consistency checks on billing datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "line_item_id",
    "bill_id",
    "hospital_id",
    "bill_date",
    "category",
    "item_description",
    "quantity",
    "billed_unit",
    "billed_unit_price",
    "billed_line_amount",
    "reference_unit_price",
    "match_status",
    "suitable_for_comparison",
]


@dataclass
class ValidationIssue:
    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    count: int
    percentage: float
    description: str
    affected_rows: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    total_records: int
    total_bills: int
    is_valid: bool
    missing_value_counts: dict[str, int]
    missing_value_rate: float
    duplicate_rows_count: int
    invalid_quantity_count: int
    arithmetic_mismatch_count: int
    non_positive_reference_count: int
    unit_mismatch_count: int
    unmatched_item_count: int
    missing_reference_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "total_bills": self.total_bills,
            "is_valid": self.is_valid,
            "missing_value_rate": round(self.missing_value_rate, 2),
            "duplicate_rows_count": self.duplicate_rows_count,
            "invalid_quantity_count": self.invalid_quantity_count,
            "arithmetic_mismatch_count": self.arithmetic_mismatch_count,
            "non_positive_reference_count": self.non_positive_reference_count,
            "unit_mismatch_count": self.unit_mismatch_count,
            "unmatched_item_count": self.unmatched_item_count,
            "missing_reference_count": self.missing_reference_count,
            "issue_summary": [
                {
                    "issue_type": iss.issue_type,
                    "severity": iss.severity,
                    "count": iss.count,
                    "pct": round(iss.percentage, 2),
                    "description": iss.description,
                }
                for iss in self.issues
            ],
        }


def validate_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Checks if all mandatory schema columns exist in the DataFrame."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def audit_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """
    Performs comprehensive data quality audit across structural integrity,
    logical consistency, and reference compatibility.
    """
    total_rows = len(df)
    if total_rows == 0:
        return DataQualityReport(
            total_records=0,
            total_bills=0,
            is_valid=False,
            missing_value_counts={},
            missing_value_rate=0.0,
            duplicate_rows_count=0,
            invalid_quantity_count=0,
            arithmetic_mismatch_count=0,
            non_positive_reference_count=0,
            unit_mismatch_count=0,
            unmatched_item_count=0,
            missing_reference_count=0,
            issues=[
                ValidationIssue(
                    issue_type="empty_dataset",
                    severity="critical",
                    count=0,
                    percentage=0.0,
                    description="Input dataset contains 0 records.",
                )
            ],
        )

    issues: list[ValidationIssue] = []

    # 1. Missing values
    missing_per_col = df.isnull().sum().to_dict()
    total_cells = total_rows * len(df.columns)
    total_nulls = sum(missing_per_col.values())
    missing_rate = (total_nulls / total_cells) * 100 if total_cells > 0 else 0.0

    # 2. Duplicate detection:
    # A true duplicate in billing analysis is identical bill_id, item_description, quantity, billed_unit_price, and bill_date
    dup_subset = ["bill_id", "item_description", "quantity", "billed_unit_price"]
    available_dup_cols = [c for c in dup_subset if c in df.columns]
    duplicate_mask = df.duplicated(subset=available_dup_cols, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        dup_ids = df.loc[duplicate_mask, "line_item_id"].head(5).tolist() if "line_item_id" in df.columns else []
        issues.append(
            ValidationIssue(
                issue_type="duplicate_line_items",
                severity="warning",
                count=duplicate_count,
                percentage=(duplicate_count / total_rows) * 100,
                description=f"Found {duplicate_count} duplicate line item entries with identical bill, item, qty, and rate.",
                affected_rows=dup_ids,
            )
        )

    # 3. Invalid quantities (<= 0 or NaN)
    invalid_qty_mask = df["quantity"].isnull() | (df["quantity"] <= 0)
    invalid_qty_count = int(invalid_qty_mask.sum())
    if invalid_qty_count > 0:
        qty_ids = df.loc[invalid_qty_mask, "line_item_id"].head(5).tolist() if "line_item_id" in df.columns else []
        issues.append(
            ValidationIssue(
                issue_type="invalid_quantity",
                severity="critical",
                count=invalid_qty_count,
                percentage=(invalid_qty_count / total_rows) * 100,
                description=f"Found {invalid_qty_count} rows with non-positive or null billing quantity.",
                affected_rows=qty_ids,
            )
        )

    # 4. Arithmetic consistency: billed_line_amount ~= quantity * billed_unit_price
    # Check only on valid positive quantities
    expected_amount = df["quantity"] * df["billed_unit_price"]
    arithmetic_err_mask = (df["quantity"] > 0) & (
        (df["billed_line_amount"] - expected_amount).abs() > 0.05
    )
    arithmetic_mismatch_count = int(arithmetic_err_mask.sum())
    if arithmetic_mismatch_count > 0:
        math_ids = df.loc[arithmetic_err_mask, "line_item_id"].head(5).tolist() if "line_item_id" in df.columns else []
        issues.append(
            ValidationIssue(
                issue_type="arithmetic_mismatch",
                severity="warning",
                count=arithmetic_mismatch_count,
                percentage=(arithmetic_mismatch_count / total_rows) * 100,
                description=f"Found {arithmetic_mismatch_count} rows where billed_line_amount differs from (quantity * billed_unit_price).",
                affected_rows=math_ids,
            )
        )

    # 5. Non-positive reference prices (reference_unit_price <= 0 while present)
    non_pos_ref_mask = df["reference_unit_price"].notnull() & (df["reference_unit_price"] <= 0)
    non_pos_ref_count = int(non_pos_ref_mask.sum())
    if non_pos_ref_count > 0:
        ref_ids = df.loc[non_pos_ref_mask, "line_item_id"].head(5).tolist() if "line_item_id" in df.columns else []
        issues.append(
            ValidationIssue(
                issue_type="non_positive_reference_price",
                severity="critical",
                count=non_pos_ref_count,
                percentage=(non_pos_ref_count / total_rows) * 100,
                description=f"Found {non_pos_ref_count} reference benchmark prices that are zero or negative.",
                affected_rows=ref_ids,
            )
        )

    # 6. Unit mismatches
    if "reference_unit" in df.columns and "billed_unit" in df.columns:
        unit_mismatch_mask = (df.get("match_status") == "unit_mismatch") | (
            df["reference_unit"].notnull()
            & df["billed_unit"].notnull()
            & (df["billed_unit"].astype(str).str.lower() != df["reference_unit"].astype(str).str.lower())
        )
    elif "match_status" in df.columns:
        unit_mismatch_mask = df["match_status"] == "unit_mismatch"
    else:
        unit_mismatch_mask = pd.Series([False] * total_rows)

    unit_mismatch_count = int(unit_mismatch_mask.sum())
    if unit_mismatch_count > 0:
        issues.append(
            ValidationIssue(
                issue_type="unit_mismatch",
                severity="warning",
                count=unit_mismatch_count,
                percentage=(unit_mismatch_count / total_rows) * 100,
                description=f"Identified {unit_mismatch_count} rows with incompatible billing vs reference units of measure.",
            )
        )

    # 7. Unmatched items & missing reference prices
    unmatched_count = int((df["match_status"] == "unmatched").sum())
    missing_ref_count = int(
        (df["match_status"] == "missing_reference").sum()
        | (df["reference_unit_price"].isnull() & (df["match_status"] != "unmatched")).sum()
    )

    if unmatched_count > 0:
        issues.append(
            ValidationIssue(
                issue_type="unmatched_items",
                severity="info",
                count=unmatched_count,
                percentage=(unmatched_count / total_rows) * 100,
                description=f"{unmatched_count} items could not be mapped to any reference catalog entry.",
            )
        )

    total_bills = df["bill_id"].nunique() if "bill_id" in df.columns else 0
    is_valid = invalid_qty_count == 0 and non_pos_ref_count == 0

    return DataQualityReport(
        total_records=total_rows,
        total_bills=total_bills,
        is_valid=is_valid,
        missing_value_counts=missing_per_col,
        missing_value_rate=missing_rate,
        duplicate_rows_count=duplicate_count,
        invalid_quantity_count=invalid_qty_count,
        arithmetic_mismatch_count=arithmetic_mismatch_count,
        non_positive_reference_count=non_pos_ref_count,
        unit_mismatch_count=unit_mismatch_count,
        unmatched_item_count=unmatched_count,
        missing_reference_count=missing_ref_count,
        issues=issues,
    )
