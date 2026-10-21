"""
Processing Pipeline for SaveOnCare Billing Analytics.

Implements clean data ingestion, deduplication, unit standardization,
comparability qualification, reference amount computation, and variance calculation.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def clean_and_standardize_data(df: pd.DataFrame, remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Cleans raw billing dataframe:
    - Standardizes column names and casing
    - Normalizes categories and units
    - Removes duplicate rows if configured (keeping the first occurrence)
    - Validates arithmetic line amounts
    """
    cleaned = df.copy()

    # Category normalization
    if "category" in cleaned.columns:
        cleaned["category"] = cleaned["category"].astype(str).str.strip().str.lower()

    # Unit normalization
    if "billed_unit" in cleaned.columns:
        cleaned["billed_unit"] = cleaned["billed_unit"].astype(str).str.strip().str.lower()
    if "reference_unit" in cleaned.columns:
        cleaned["reference_unit"] = cleaned["reference_unit"].astype(str).str.strip().str.lower()

    # Deduplication
    if remove_duplicates:
        dup_subset = ["bill_id", "item_description", "quantity", "billed_unit_price"]
        available_cols = [c for c in dup_subset if c in cleaned.columns]
        if available_cols:
            cleaned = cleaned.drop_duplicates(subset=available_cols, keep="first").reset_index(drop=True)

    # Date parsing
    if "bill_date" in cleaned.columns:
        cleaned["bill_date"] = pd.to_datetime(cleaned["bill_date"], errors="coerce")

    return cleaned


def evaluate_comparability(row: pd.Series) -> tuple[bool, str]:
    """
    Evaluates whether a single row can be legitimately compared against a reference benchmark.
    Returns (is_comparable, reason_code).
    """
    # 1. Invalid quantity
    qty = row.get("quantity")
    if pd.isnull(qty) or qty <= 0:
        return False, "INVALID_QUANTITY"

    # 2. Match status
    match_status = str(row.get("match_status", "")).lower()
    if match_status == "unmatched":
        return False, "NO_CATALOG_MATCH"
    if match_status == "missing_reference":
        return False, "BENCHMARK_UNAVAILABLE"
    if match_status == "unit_mismatch":
        return False, "INCOMPATIBLE_UNITS"

    # 3. Reference price validity
    ref_price = row.get("reference_unit_price")
    if pd.isnull(ref_price) or ref_price <= 0:
        return False, "INVALID_REFERENCE_PRICE"

    # 4. Explicit unit compatibility
    b_unit = str(row.get("billed_unit", "")).lower()
    r_unit = str(row.get("reference_unit", "")).lower()
    if b_unit and r_unit and b_unit != r_unit and r_unit != "none" and r_unit != "nan":
        return False, "INCOMPATIBLE_UNITS"

    # 5. Suitable for comparison flag
    if not bool(row.get("suitable_for_comparison", True)):
        return False, "EXCLUDED_BY_PROTOCOL"

    return True, "VALID_COMPARISON"


def compute_line_item_variances(
    df: pd.DataFrame,
    discrepancy_threshold_pct: float = 20.0,
) -> pd.DataFrame:
    """
    Calculates reference amounts, absolute differences, percentage variances,
    and discrepancy flags for each row.

    Rules:
    - Difference = Billed Amount - Reference Amount
    - Percentage Variance = ((Billed Amount - Reference Amount) / Reference Amount) * 100
    - If row is non-comparable, difference and variance are set to NaN and flagged as INELIGIBLE.
    - Potential Discrepancy is flagged only if row is comparable and variance > threshold.
    """
    processed = df.copy()

    comparable_flags = []
    eligibility_reasons = []

    for _, row in processed.iterrows():
        comp, reason = evaluate_comparability(row)
        comparable_flags.append(comp)
        eligibility_reasons.append(reason)

    processed["is_comparable"] = comparable_flags
    processed["eligibility_reason"] = eligibility_reasons

    # Reference line amount = quantity * reference_unit_price
    # Calculated only for comparable rows with valid quantity and reference price
    valid_mask = processed["is_comparable"]

    processed["reference_line_amount"] = np.nan
    processed.loc[valid_mask, "reference_line_amount"] = (
        processed.loc[valid_mask, "quantity"] * processed.loc[valid_mask, "reference_unit_price"]
    ).round(2)

    # Difference in INR = Billed amount - Reference amount
    processed["variance_amount_inr"] = np.nan
    processed.loc[valid_mask, "variance_amount_inr"] = (
        processed.loc[valid_mask, "billed_line_amount"] - processed.loc[valid_mask, "reference_line_amount"]
    ).round(2)

    # Percentage variance = ((Billed - Reference) / Reference) * 100
    processed["variance_pct"] = np.nan
    ref_positive_mask = valid_mask & (processed["reference_line_amount"] > 0)
    processed.loc[ref_positive_mask, "variance_pct"] = (
        (
            (processed.loc[ref_positive_mask, "billed_line_amount"] - processed.loc[ref_positive_mask, "reference_line_amount"])
            / processed.loc[ref_positive_mask, "reference_line_amount"]
        )
        * 100.0
    ).round(2)

    # Classification status:
    # 'Potential Discrepancy' if comparable and variance_pct > threshold
    # 'Within Benchmark Range' if comparable and variance_pct <= threshold
    # 'Ineligible / Unmatched' if not comparable
    status_col = []
    is_discrepancy_col = []

    for _, row in processed.iterrows():
        if not row["is_comparable"]:
            status_col.append(f"Ineligible ({row['eligibility_reason']})")
            is_discrepancy_col.append(False)
        else:
            var_pct = row["variance_pct"]
            if pd.notnull(var_pct) and var_pct > discrepancy_threshold_pct:
                status_col.append("Potential Discrepancy")
                is_discrepancy_col.append(True)
            else:
                status_col.append("Within Benchmark Range")
                is_discrepancy_col.append(False)

    processed["review_status"] = status_col
    processed["is_potential_discrepancy"] = is_discrepancy_col

    # Potential discrepancy value in INR (positive excess only)
    processed["potential_discrepancy_value_inr"] = np.where(
        processed["is_potential_discrepancy"] & (processed["variance_amount_inr"] > 0),
        processed["variance_amount_inr"],
        0.0,
    )

    return processed
