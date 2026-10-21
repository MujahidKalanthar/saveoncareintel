"""
Unit tests for SaveOnCare data validation and quality auditing.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation import audit_data_quality, validate_required_columns


def test_validate_required_columns():
    """Tests schema column presence checks."""
    valid_df = pd.DataFrame(
        columns=[
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
    )
    is_valid, missing = validate_required_columns(valid_df)
    assert is_valid == True
    assert len(missing) == 0

    invalid_df = pd.DataFrame(columns=["line_item_id", "category"])
    is_valid, missing = validate_required_columns(invalid_df)
    assert is_valid == False
    assert "bill_id" in missing


def test_duplicate_row_detection():
    """Tests identification of duplicate line items."""
    df = pd.DataFrame(
        [
            {
                "line_item_id": "L1",
                "bill_id": "B1",
                "item_description": "Syringe 10ml",
                "quantity": 5,
                "billed_unit": "piece",
                "billed_unit_price": 10.0,
                "billed_line_amount": 50.0,
                "reference_unit_price": 8.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            {
                "line_item_id": "L2",
                "bill_id": "B1",
                "item_description": "Syringe 10ml",
                "quantity": 5,
                "billed_unit": "piece",
                "billed_unit_price": 10.0,
                "billed_line_amount": 50.0,
                "reference_unit_price": 8.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
        ]
    )
    report = audit_data_quality(df)
    assert report.duplicate_rows_count == 1
    issue_types = [iss.issue_type for iss in report.issues]
    assert "duplicate_line_items" in issue_types


def test_invalid_quantity_and_arithmetic_checks():
    """Tests detection of zero/negative quantities and calculation errors."""
    df = pd.DataFrame(
        [
            # Zero quantity
            {
                "line_item_id": "L1",
                "bill_id": "B1",
                "item_description": "Gauze Pack",
                "quantity": 0,
                "billed_unit": "pack",
                "billed_unit_price": 50.0,
                "billed_line_amount": 0.0,
                "reference_unit_price": 40.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            # Negative quantity
            {
                "line_item_id": "L2",
                "bill_id": "B1",
                "item_description": "Gauze Pack",
                "quantity": -2,
                "billed_unit": "pack",
                "billed_unit_price": 50.0,
                "billed_line_amount": -100.0,
                "reference_unit_price": 40.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            # Arithmetic mismatch: qty 2 * price 50 should be 100, but billed 150
            {
                "line_item_id": "L3",
                "bill_id": "B1",
                "item_description": "Gauze Pack",
                "quantity": 2,
                "billed_unit": "pack",
                "billed_unit_price": 50.0,
                "billed_line_amount": 150.0,
                "reference_unit_price": 40.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
        ]
    )
    report = audit_data_quality(df)
    assert report.invalid_quantity_count == 2
    assert report.arithmetic_mismatch_count == 1


def test_non_positive_reference_prices():
    """Tests detection of reference benchmark prices <= 0."""
    df = pd.DataFrame(
        [
            {
                "line_item_id": "L1",
                "bill_id": "B1",
                "item_description": "Item Zero",
                "quantity": 1,
                "billed_unit": "vial",
                "billed_unit_price": 100.0,
                "billed_line_amount": 100.0,
                "reference_unit_price": 0.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            {
                "line_item_id": "L2",
                "bill_id": "B1",
                "item_description": "Item Neg",
                "quantity": 1,
                "billed_unit": "vial",
                "billed_unit_price": 100.0,
                "billed_line_amount": 100.0,
                "reference_unit_price": -25.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
        ]
    )
    report = audit_data_quality(df)
    assert report.non_positive_reference_count == 2
    assert report.is_valid == False


def test_unit_mismatch_detection():
    """Tests detection of incompatible billed vs reference units."""
    df = pd.DataFrame(
        [
            {
                "line_item_id": "L1",
                "bill_id": "B1",
                "item_description": "Syringes Box",
                "quantity": 1,
                "billed_unit": "box_of_100",
                "billed_unit_price": 500.0,
                "billed_line_amount": 500.0,
                "reference_unit": "piece",
                "reference_unit_price": 6.0,
                "match_status": "unit_mismatch",
                "suitable_for_comparison": False,
            }
        ]
    )
    report = audit_data_quality(df)
    assert report.unit_mismatch_count == 1
