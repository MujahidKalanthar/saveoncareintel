"""
Unit tests for SaveOnCare metrics calculations and financial modeling.
"""

import numpy as np
import pandas as pd
import pytest

from src.business_model import run_pilot_economics_simulation
from src.metrics import (
    calculate_category_summary,
    calculate_executive_metrics,
    evaluate_test_dataset_performance,
)
from src.processing import compute_line_item_variances


def test_difference_and_percentage_variance_calculation():
    """Validates basic difference and percentage variance calculations."""
    sample_data = pd.DataFrame(
        [
            {
                "line_item_id": "L1",
                "bill_id": "B1",
                "hospital_id": "H1",
                "category": "medicines",
                "item_description": "Paracetamol 500mg",
                "quantity": 2,
                "billed_unit": "vial",
                "billed_unit_price": 60.0,
                "billed_line_amount": 120.0,
                "reference_unit": "vial",
                "reference_unit_price": 50.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            {
                "line_item_id": "L2",
                "bill_id": "B1",
                "hospital_id": "H1",
                "category": "medicines",
                "item_description": "Normal Saline",
                "quantity": 1,
                "billed_unit": "bottle",
                "billed_unit_price": 50.0,
                "billed_line_amount": 50.0,
                "reference_unit": "bottle",
                "reference_unit_price": 50.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
        ]
    )

    processed = compute_line_item_variances(sample_data, discrepancy_threshold_pct=15.0)

    # L1: Reference amount = 2 * 50 = 100. Diff = 120 - 100 = 20. Variance = (20/100)*100 = 20%.
    # Variance 20% > 15% threshold -> flagged as potential discrepancy.
    row1 = processed.loc[processed["line_item_id"] == "L1"].iloc[0]
    assert row1["reference_line_amount"] == 100.0
    assert row1["variance_amount_inr"] == 20.0
    assert row1["variance_pct"] == 20.0
    assert row1["is_potential_discrepancy"] == True
    assert row1["potential_discrepancy_value_inr"] == 20.0

    # L2: Reference amount = 1 * 50 = 50. Diff = 0. Variance = 0%.
    row2 = processed.loc[processed["line_item_id"] == "L2"].iloc[0]
    assert row2["reference_line_amount"] == 50.0
    assert row2["variance_amount_inr"] == 0.0
    assert row2["variance_pct"] == 0.0
    assert row2["is_potential_discrepancy"] == False
    assert row2["potential_discrepancy_value_inr"] == 0.0


def test_benchmark_coverage_and_discrepancy_rate():
    """Tests item coverage, value coverage, and potential discrepancy rate."""
    sample_data = pd.DataFrame(
        [
            # Comparable item 1: Billed 100, Ref 80
            {
                "line_item_id": "L1",
                "bill_id": "B1",
                "category": "consumables",
                "item_description": "Item 1",
                "quantity": 1,
                "billed_unit": "piece",
                "billed_unit_price": 100.0,
                "billed_line_amount": 100.0,
                "reference_unit": "piece",
                "reference_unit_price": 80.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            # Comparable item 2: Billed 100, Ref 100
            {
                "line_item_id": "L2",
                "bill_id": "B1",
                "category": "consumables",
                "item_description": "Item 2",
                "quantity": 1,
                "billed_unit": "piece",
                "billed_unit_price": 100.0,
                "billed_line_amount": 100.0,
                "reference_unit": "piece",
                "reference_unit_price": 100.0,
                "match_status": "matched",
                "suitable_for_comparison": True,
            },
            # Incomparable item 3: Unmatched, Billed 200
            {
                "line_item_id": "L3",
                "bill_id": "B2",
                "category": "consumables",
                "item_description": "Item 3",
                "quantity": 1,
                "billed_unit": "piece",
                "billed_unit_price": 200.0,
                "billed_line_amount": 200.0,
                "reference_unit": None,
                "reference_unit_price": np.nan,
                "match_status": "unmatched",
                "suitable_for_comparison": False,
            },
            # Incomparable item 4: Missing reference, Billed 100
            {
                "line_item_id": "L4",
                "bill_id": "B2",
                "category": "consumables",
                "item_description": "Item 4",
                "quantity": 1,
                "billed_unit": "piece",
                "billed_unit_price": 100.0,
                "billed_line_amount": 100.0,
                "reference_unit": "piece",
                "reference_unit_price": np.nan,
                "match_status": "missing_reference",
                "suitable_for_comparison": False,
            },
        ]
    )

    processed = compute_line_item_variances(sample_data, discrepancy_threshold_pct=20.0)
    metrics = calculate_executive_metrics(processed, threshold_pct=20.0)

    # Total items = 4. Comparable = 2.
    assert metrics.total_line_items == 4
    assert metrics.comparable_line_items == 2
    assert metrics.item_benchmark_coverage_pct == 50.0  # 2 / 4 = 50%

    # Total Billed = 500. Comparable Billed = 200.
    assert metrics.total_billed_amount_inr == 500.0
    assert metrics.comparable_billed_amount_inr == 200.0
    assert metrics.value_benchmark_coverage_pct == 40.0  # 200 / 500 = 40%

    # L1: Billed 100, Ref 80 -> Variance = (20/80)*100 = 25% > 20% -> Flagged.
    # L2: Billed 100, Ref 100 -> Variance = 0% -> Not flagged.
    # Discrepancy count = 1. Comparable count = 2. Rate = 50%.
    assert metrics.potential_discrepancy_count == 1
    assert metrics.potential_discrepancy_rate_pct == 50.0
    assert metrics.potential_discrepancy_value_inr == 20.0


def test_test_dataset_validation_metrics():
    """Validates confusion matrix and precision/recall calculations."""
    processed = pd.DataFrame(
        [
            # TP: predicted True, true True
            {"is_comparable": True, "is_potential_discrepancy": True, "injected_discrepancy": True},
            # FP: predicted True, true False
            {"is_comparable": True, "is_potential_discrepancy": True, "injected_discrepancy": False},
            # FN: predicted False, true True
            {"is_comparable": True, "is_potential_discrepancy": False, "injected_discrepancy": True},
            # TN: predicted False, true False
            {"is_comparable": True, "is_potential_discrepancy": False, "injected_discrepancy": False},
            # Incomparable row should be excluded from evaluation
            {"is_comparable": False, "is_potential_discrepancy": False, "injected_discrepancy": True},
        ]
    )

    res = evaluate_test_dataset_performance(processed)
    assert res.true_positives == 1
    assert res.false_positives == 1
    assert res.false_negatives == 1
    assert res.true_negatives == 1
    assert res.precision == 50.0  # 1 / (1 + 1) = 50%
    assert res.recall == 50.0     # 1 / (1 + 1) = 50%
    assert res.f1_score == 50.0


def test_financial_calculations():
    """Tests pilot economics calculations and break-even arithmetic."""
    # 1000 bills @ 300 INR fee = 300,000 INR
    # Setup fee = 50,000 INR -> Total Revenue = 350,000 INR
    # Variable cost = 100 INR/bill -> Total Var Cost = 100,000 INR
    # Monthly fixed ops = 20,000 INR * 5 months = 100,000 INR
    # Total Modelled Costs = 200,000 INR
    # Operating Surplus = 350,000 - 200,000 = 150,000 INR
    res = run_pilot_economics_simulation(
        bills_processed=1000,
        contract_price_per_bill_inr=300.0,
        setup_fee_inr=50000.0,
        variable_cost_per_bill_inr=100.0,
        monthly_support_ops_inr=20000.0,
        pilot_duration_months=5,
    )

    assert res.assumed_revenue_inr == 350000.0
    assert res.variable_costs_inr == 100000.0
    assert res.fixed_operating_costs_inr == 100000.0
    assert res.total_modelled_costs_inr == 200000.0
    assert res.net_operating_surplus_inr == 150000.0
    assert res.gross_contribution_inr == 250000.0

    # Unit contribution = 300 - 100 = 200 INR.
    # Fixed burden = 100,000 - 50,000 = 50,000 INR.
    # Break-even bills = 50,000 / 200 = 250 bills.
    assert res.break_even_bills == 250.0
    assert res.break_even_achieved == True
