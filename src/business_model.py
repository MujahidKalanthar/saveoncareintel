"""
Pilot Economics and Business Intelligence Modeling for SaveOnCare by VoltEdge.

Models unit economics, revenue mechanics, cost structures, contribution margins,
break-even thresholds, and multi-scenario sensitivities for enterprise pilots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class PilotEconomicsResult:
    # Inputs
    bills_processed: int
    contract_price_per_bill_inr: float
    setup_fee_inr: float
    variable_cost_per_bill_inr: float
    monthly_support_ops_inr: float
    pilot_duration_months: int

    # Financial Outputs
    assumed_revenue_inr: float
    variable_costs_inr: float
    fixed_operating_costs_inr: float
    total_modelled_costs_inr: float
    gross_contribution_inr: float
    contribution_margin_pct: float
    net_operating_surplus_inr: float
    cost_per_processed_bill_inr: float
    break_even_bills: float | None
    break_even_achieved: bool

    # Sensitivity & Elasticity Ranking
    elasticity_ranking: list[dict[str, Any]]
    disclaimer: str


def run_pilot_economics_simulation(
    bills_processed: int = 1500,
    contract_price_per_bill_inr: float = 350.0,
    setup_fee_inr: float = 50000.0,
    variable_cost_per_bill_inr: float = 120.0,
    monthly_support_ops_inr: float = 25000.0,
    pilot_duration_months: int = 6,
) -> PilotEconomicsResult:
    """
    Computes pilot financial model with rigorous accounting distinctions:
    - Assumed Revenue = (Bills * Contract Fee) + Setup Fee
    - Variable Costs = Bills * Variable Cost Per Bill (data parsing, OCR, clinician spot-check)
    - Fixed Operating Costs = Monthly Support * Duration
    - Total Modelled Costs = Variable Costs + Fixed Operating Costs
    - Gross Contribution = Assumed Revenue - Variable Costs
    - Net Operating Surplus = Assumed Revenue - Total Modelled Costs
    - Break-Even Volume = Fixed Operating Costs / (Contract Price - Variable Cost) (when price > var_cost)
    """
    # 1. Revenue
    bill_revenue = bills_processed * contract_price_per_bill_inr
    total_revenue = bill_revenue + setup_fee_inr

    # 2. Costs
    total_var_cost = bills_processed * variable_cost_per_bill_inr
    total_fixed_cost = monthly_support_ops_inr * pilot_duration_months
    total_modelled_costs = total_var_cost + total_fixed_cost

    # 3. Margins
    unit_contribution = contract_price_per_bill_inr - variable_cost_per_bill_inr
    gross_contribution = total_revenue - total_var_cost
    contribution_margin_pct = (gross_contribution / total_revenue) * 100.0 if total_revenue > 0 else 0.0

    net_operating_surplus = total_revenue - total_modelled_costs
    cost_per_bill = total_modelled_costs / bills_processed if bills_processed > 0 else 0.0

    # 4. Break-even calculation (amortizing fixed costs + setup offset)
    net_fixed_burden = total_fixed_cost - setup_fee_inr
    if unit_contribution > 0:
        if net_fixed_burden <= 0:
            break_even_bills = 0.0
            break_even_achieved = True
        else:
            break_even_bills = net_fixed_burden / unit_contribution
            break_even_achieved = bills_processed >= break_even_bills
    else:
        # Negative unit economics: can never break even
        break_even_bills = None
        break_even_achieved = False

    # 5. Sensitivity & Elasticity analysis:
    # Measure % change in Net Operating Surplus for a +10% increase in each input parameter
    base_surplus = net_operating_surplus if net_operating_surplus != 0 else 1.0

    # Sensitivity to +10% price
    surplus_price = (
        (bills_processed * contract_price_per_bill_inr * 1.10 + setup_fee_inr) - total_modelled_costs
    )
    sens_price = abs((surplus_price - net_operating_surplus) / base_surplus) * 100

    # Sensitivity to +10% volume
    surplus_vol = (
        ((bills_processed * 1.10) * contract_price_per_bill_inr + setup_fee_inr)
        - ((bills_processed * 1.10) * variable_cost_per_bill_inr + total_fixed_cost)
    )
    sens_vol = abs((surplus_vol - net_operating_surplus) / base_surplus) * 100

    # Sensitivity to +10% variable cost
    surplus_var = total_revenue - (bills_processed * variable_cost_per_bill_inr * 1.10 + total_fixed_cost)
    sens_var = abs((surplus_var - net_operating_surplus) / base_surplus) * 100

    # Sensitivity to +10% fixed cost
    surplus_fix = total_revenue - (total_var_cost + total_fixed_cost * 1.10)
    sens_fix = abs((surplus_fix - net_operating_surplus) / base_surplus) * 100

    elasticity = [
        {"Parameter": "Contract Price Per Bill", "Impact Score (%)": round(sens_price, 1), "Type": "Direct Revenue"},
        {"Parameter": "Processed Bill Volume", "Impact Score (%)": round(sens_vol, 1), "Type": "Scale Driver"},
        {"Parameter": "Variable Processing Cost", "Impact Score (%)": round(sens_var, 1), "Type": "Efficiency"},
        {"Parameter": "Fixed Support & Ops", "Impact Score (%)": round(sens_fix, 1), "Type": "Overhead"},
    ]
    elasticity.sort(key=lambda x: x["Impact Score (%)"], reverse=True)

    disclaimer = (
        "ILLUSTRATIVE ASSUMPTION MODEL ONLY: Financial estimates do not represent actual institutional "
        "contracts, audited financial statements, or committed commercial pricing. Cash flow timing, working capital, "
        "taxes, and sales acquisition cycles are excluded."
    )

    return PilotEconomicsResult(
        bills_processed=bills_processed,
        contract_price_per_bill_inr=contract_price_per_bill_inr,
        setup_fee_inr=setup_fee_inr,
        variable_cost_per_bill_inr=variable_cost_per_bill_inr,
        monthly_support_ops_inr=monthly_support_ops_inr,
        pilot_duration_months=pilot_duration_months,
        assumed_revenue_inr=round(total_revenue, 2),
        variable_costs_inr=round(total_var_cost, 2),
        fixed_operating_costs_inr=round(total_fixed_cost, 2),
        total_modelled_costs_inr=round(total_modelled_costs, 2),
        gross_contribution_inr=round(gross_contribution, 2),
        contribution_margin_pct=round(contribution_margin_pct, 1),
        net_operating_surplus_inr=round(net_operating_surplus, 2),
        cost_per_processed_bill_inr=round(cost_per_bill, 2),
        break_even_bills=round(break_even_bills, 1) if break_even_bills is not None else None,
        break_even_achieved=break_even_achieved,
        elasticity_ranking=elasticity,
        disclaimer=disclaimer,
    )


def generate_sensitivity_matrix(
    base_result: PilotEconomicsResult,
    volume_multipliers: list[float] | None = None,
    price_multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """
    Generates a 2D sensitivity matrix of Net Operating Surplus (₹)
    across variations in Bill Volume and Contract Price.
    """
    if volume_multipliers is None:
        volume_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    if price_multipliers is None:
        price_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]

    base_vol = base_result.bills_processed
    base_price = base_result.contract_price_per_bill_inr

    matrix_data: dict[str, list[float]] = {}

    for p_mult in price_multipliers:
        price_val = base_price * p_mult
        col_label = f"Price: ₹{int(price_val)} ({(p_mult-1)*100:+.0f}%)"
        row_vals = []

        for v_mult in volume_multipliers:
            vol_val = int(base_vol * v_mult)
            sim = run_pilot_economics_simulation(
                bills_processed=vol_val,
                contract_price_per_bill_inr=price_val,
                setup_fee_inr=base_result.setup_fee_inr,
                variable_cost_per_bill_inr=base_result.variable_cost_per_bill_inr,
                monthly_support_ops_inr=base_result.monthly_support_ops_inr,
                pilot_duration_months=base_result.pilot_duration_months,
            )
            row_vals.append(sim.net_operating_surplus_inr)

        matrix_data[col_label] = row_vals

    index_labels = [f"Vol: {int(base_vol * v)} ({(v-1)*100:+.0f}%)" for v in volume_multipliers]
    return pd.DataFrame(matrix_data, index=index_labels)
