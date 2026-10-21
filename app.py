"""
SaveOnCare Intelligence Lab — Healthcare Billing Analytics & Business Intelligence
Built for VoltEdge Healthcare Solutions.
"""

from __future__ import annotations

import datetime
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.business_model import (
    generate_sensitivity_matrix,
    run_pilot_economics_simulation,
)
from src.data_generation import generate_synthetic_dataset, save_synthetic_dataset
from src.metrics import (
    calculate_category_summary,
    calculate_executive_metrics,
    evaluate_test_dataset_performance,
)
from src.processing import clean_and_standardize_data, compute_line_item_variances
from src.validation import audit_data_quality, validate_required_columns
from src.visualisations import (
    plot_benchmark_coverage_breakdown,
    plot_break_even_curve,
    plot_category_billed_vs_discrepancy,
    plot_confusion_matrix_heatmap,
    plot_monthly_trend,
    plot_sensitivity_heatmap,
    plot_variance_distribution,
)

# Page configuration
st.set_page_config(
    page_title="SaveOnCare Intelligence Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        color: #92400E;
        font-size: 0.92rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F766E;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .badge-flag {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-match {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ineligible {
        background-color: #F1F5F9;
        color: #475569;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(file_path: str = "data/synthetic_billing_data.csv") -> pd.DataFrame:
    """Loads or generates synthetic billing dataset."""
    path = Path(file_path)
    if not path.exists():
        return save_synthetic_dataset(str(path))
    return pd.read_csv(file_path)


# Ingest data
raw_df = load_data()

# -------------------------------------------------------------
# GLOBAL SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=400&auto=format&fit=crop&q=80",
        use_container_width=True,
    )
    st.markdown("### **SaveOnCare Intelligence Lab**")
    st.caption("VoltEdge Healthcare Billing Transparency Analytics")

    st.markdown("---")
    st.subheader("⚙️ Analytical Parameters")

    discrepancy_threshold = st.slider(
        "Discrepancy Review Threshold (%)",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Percentage markup above benchmark reference at which a line item is classified for potential discrepancy review.",
    )

    deduplicate_data = st.checkbox(
        "Auto-Deduplicate Identical Rows",
        value=True,
        help="Detects and removes identical duplicate bill lines before computing variances to avoid double counting.",
    )

    st.markdown("---")
    st.subheader("🔍 Cohort Filters")

    # Categories filter
    all_categories = sorted(raw_df["category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Category Filter",
        options=all_categories,
        default=all_categories,
        help="Filter line items by healthcare service domain.",
    )

    # Hospitals filter
    all_hospitals = sorted(raw_df["hospital_id"].dropna().unique().tolist())
    selected_hospitals = st.multiselect(
        "Hospital ID Filter",
        options=all_hospitals,
        default=all_hospitals,
    )

    # Match status filter
    all_match_statuses = sorted(raw_df["match_status"].dropna().unique().tolist())
    selected_statuses = st.multiselect(
        "Benchmark Match Status",
        options=all_match_statuses,
        default=all_match_statuses,
    )

    # Date range
    raw_df["bill_date_parsed"] = pd.to_datetime(raw_df["bill_date"], errors="coerce")
    min_date = raw_df["bill_date_parsed"].min().date() if not raw_df["bill_date_parsed"].isnull().all() else datetime.date(2024, 1, 1)
    max_date = raw_df["bill_date_parsed"].max().date() if not raw_df["bill_date_parsed"].isnull().all() else datetime.date(2024, 12, 31)

    date_range = st.date_input(
        "Bill Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")
    if st.button("🔄 Regenerate Synthetic Data"):
        save_synthetic_dataset()
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748B; margin-top:1.5rem; line-height:1.4;">
        <b>Notice:</b> Demonstration model using synthetic test records. All benchmark prices are illustrative medians.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# FILTERING & DATA PROCESSING PIPELINE
# -------------------------------------------------------------
# 1. Clean & standardize
cleaned_df = clean_and_standardize_data(raw_df, remove_duplicates=deduplicate_data)

# 2. Filter application
filtered_df = cleaned_df.copy()

if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

if selected_hospitals:
    filtered_df = filtered_df[filtered_df["hospital_id"].isin(selected_hospitals)]

if selected_statuses:
    filtered_df = filtered_df[filtered_df["match_status"].isin(selected_statuses)]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["bill_date"].dt.date >= start_d) & (filtered_df["bill_date"].dt.date <= end_d)
    ]

# 3. Calculate variances and flags
processed_df = compute_line_item_variances(filtered_df, discrepancy_threshold_pct=float(discrepancy_threshold))

# 4. Compute core metrics
exec_metrics = calculate_executive_metrics(processed_df, threshold_pct=float(discrepancy_threshold))
category_summary = calculate_category_summary(processed_df)
dq_report = audit_data_quality(raw_df)  # Audit full raw file for structural transparency

# -------------------------------------------------------------
# NAVIGATION TABS
# -------------------------------------------------------------
st.markdown('<div class="main-header">🏥 SaveOnCare Intelligence Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Healthcare Billing Transparency & Analytical Benchmarking Platform · VoltEdge</div>',
    unsafe_allow_html=True,
)

# Mandatory Synthetic Data Notice
st.markdown(
    """
    <div class="disclaimer-banner">
        <b>⚠️ SYNTHETIC DEMONSTRATION DATASET:</b>
        All patient bills, line items, hospital IDs, and pricing schedules displayed here are synthetically generated
        for algorithm evaluation and business intelligence simulation. Benchmark prices represent illustrative market medians,
        <b>not</b> official NPPA statutory caps or government-notified tariffs. Injected test discrepancies are controlled anomalies
        and must not be interpreted as actual medical fraud or proven overcharges.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_exec, tab_details, tab_dq, tab_insights, tab_economics, tab_audit = st.tabs(
    [
        "📊 1. Executive Overview",
        "📑 2. Detailed Billing Analysis",
        "🛡️ 3. Data Quality & Validation",
        "💡 4. Business Insights & Ops",
        "📈 5. Pilot Economics Simulator",
        "📥 6. Export & Audit Trail",
    ]
)

# =============================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================
with tab_exec:
    st.subheader("Executive KPI Summary")
    st.caption(
        f"Aggregated across **{exec_metrics.total_bills:,}** synthetic bills and **{exec_metrics.total_line_items:,}** line items "
        f"at a review threshold of **+{discrepancy_threshold}%** above reference benchmark."
    )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Billed Amount</div>
                <div class="metric-value">₹{exec_metrics.total_billed_amount_inr:,.0f}</div>
                <div class="metric-sub">{exec_metrics.total_bills:,} bills · {exec_metrics.total_line_items:,} lines</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Benchmark Coverage</div>
                <div class="metric-value">{exec_metrics.item_benchmark_coverage_pct:.1f}%</div>
                <div class="metric-sub">{exec_metrics.value_benchmark_coverage_pct:.1f}% of billed value covered</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Potential Discrepancy Value</div>
                <div class="metric-value" style="color:#E11D48;">₹{exec_metrics.potential_discrepancy_value_inr:,.0f}</div>
                <div class="metric-sub">{exec_metrics.discrepancy_share_of_billed_pct:.1f}% of total billed amount</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Discrepancy Rate</div>
                <div class="metric-value" style="color:#E11D48;">{exec_metrics.potential_discrepancy_rate_pct:.1f}%</div>
                <div class="metric-sub">{exec_metrics.potential_discrepancy_count:,} items flagged for review</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Weighted Variance</div>
                <div class="metric-value">+{exec_metrics.amount_weighted_variance_pct:.1f}%</div>
                <div class="metric-sub">Item-weighted: +{exec_metrics.average_variance_pct_comparable:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_cat = plot_category_billed_vs_discrepancy(category_summary)
        st.plotly_chart(fig_cat, use_container_width=True)

    with chart_col2:
        fig_dist = plot_variance_distribution(processed_df, threshold_pct=float(discrepancy_threshold))
        st.plotly_chart(fig_dist, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig_trend = plot_monthly_trend(processed_df)
        st.plotly_chart(fig_trend, use_container_width=True)

    with chart_col4:
        fig_cov = plot_benchmark_coverage_breakdown(category_summary)
        st.plotly_chart(fig_cov, use_container_width=True)

    # Category Summary Table
    st.subheader("Category-Level Benchmark Coverage & Discrepancy Breakdown")
    st.dataframe(
        category_summary.style.format(
            {
                "Total Billed (₹)": "₹{:,.2f}",
                "Item Coverage (%)": "{:.1f}%",
                "Value Coverage (%)": "{:.1f}%",
                "Discrepancy Rate (%)": "{:.1f}%",
                "Potential Discrepancy Value (₹)": "₹{:,.2f}",
                "Avg Variance (%)": "{:+.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================
# TAB 2: DETAILED BILLING ANALYSIS & EVIDENCE INSPECTOR
# =============================================================
with tab_details:
    st.subheader("Line-Item Billing Explorer & Evidence Audit")
    st.caption("Inspect individual line-item calculations, clinical descriptions, benchmark sources, and comparability rationale.")

    # Status summary counters
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.info(f"**Total Records:** {len(processed_df):,}")
    with col_s2:
        st.success(f"**Within Benchmark:** {(processed_df['review_status'] == 'Within Benchmark Range').sum():,}")
    with col_s3:
        st.error(f"**Flagged Discrepancies:** {(processed_df['is_potential_discrepancy'] == True).sum():,}")
    with col_s4:
        st.warning(f"**Ineligible / Unmatched:** {(processed_df['is_comparable'] == False).sum():,}")

    # Search bar & sort
    search_query = st.text_input("🔎 Search by Item Description or Product ID", "")
    display_df = processed_df.copy()
    if search_query:
        display_df = display_df[
            display_df["item_description"].str.contains(search_query, case=False, na=False)
            | display_df["product_or_service_id"].str.contains(search_query, case=False, na=False)
            | display_df["bill_id"].str.contains(search_query, case=False, na=False)
        ]

    # Display columns
    table_cols = [
        "line_item_id",
        "bill_id",
        "hospital_id",
        "category",
        "item_description",
        "quantity",
        "billed_unit",
        "billed_unit_price",
        "billed_line_amount",
        "reference_unit_price",
        "reference_source",
        "variance_amount_inr",
        "variance_pct",
        "review_status",
    ]
    present_table_cols = [c for c in table_cols if c in display_df.columns]

    st.dataframe(
        display_df[present_table_cols].head(300),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.markdown("---")
    st.subheader("🔬 Deep-Dive Evidence Inspector")
    st.caption("Select a specific line item ID to inspect the full audit chain, benchmark provenance, and comparison validity.")

    available_item_ids = display_df["line_item_id"].tolist()
    if available_item_ids:
        # Prioritize showing an injected discrepancy by default for demo
        flagged_sample = display_df[display_df["is_potential_discrepancy"] == True]
        default_idx = (
            available_item_ids.index(flagged_sample.iloc[0]["line_item_id"])
            if not flagged_sample.empty and flagged_sample.iloc[0]["line_item_id"] in available_item_ids
            else 0
        )

        selected_item_id = st.selectbox(
            "Select Line Item ID to Audit",
            options=available_item_ids,
            index=default_idx,
        )

        item_row = display_df[display_df["line_item_id"] == selected_item_id].iloc[0]

        # Inspector Card Layout
        card_col1, card_col2, card_col3 = st.columns([1.2, 1.2, 1.6])

        with card_col1:
            st.markdown("##### 🏥 Billed Information")
            st.write(f"**Bill ID:** `{item_row['bill_id']}`")
            st.write(f"**Hospital:** {item_row.get('hospital_name', item_row['hospital_id'])}")
            st.write(f"**Date:** {item_row['bill_date']}")
            st.write(f"**Category:** `{item_row['category'].upper()}`")
            st.write(f"**Item:** {item_row['item_description']}")
            st.write(f"**Product Code:** `{item_row.get('product_or_service_id', 'N/A')}`")
            st.write(f"**Quantity Billed:** {item_row['quantity']} {item_row['billed_unit']}")
            st.write(f"**Billed Unit Price:** ₹{item_row['billed_unit_price']:,.2f}")
            st.write(f"**Total Billed Amount:** **₹{item_row['billed_line_amount']:,.2f}**")

        with card_col2:
            st.markdown("##### ⚖️ Reference Benchmark")
            ref_pr = item_row.get("reference_unit_price")
            if pd.notnull(ref_pr):
                st.write(f"**Benchmark Unit Price:** ₹{ref_pr:,.2f}")
                st.write(f"**Benchmark Unit:** `{item_row.get('reference_unit', 'N/A')}`")
                st.write(f"**Benchmark Line Amount:** ₹{item_row.get('reference_line_amount', 0.0):,.2f}")
                st.write(f"**Benchmark Source:** {item_row.get('reference_source', 'None')}")
                st.write(f"**Reference Effective Date:** {item_row.get('reference_date', 'N/A')}")
                st.write(f"**Match Status:** `{item_row.get('match_status')}`")
            else:
                st.warning("⚠️ No valid benchmark reference available for this item in catalog.")
                st.write(f"**Match Status:** `{item_row.get('match_status')}`")

        with card_col3:
            st.markdown("##### 📋 Audit Decision & Arithmetic")
            is_comp = item_row.get("is_comparable", False)
            if is_comp:
                var_inr = item_row.get("variance_amount_inr", 0.0)
                var_pct = item_row.get("variance_pct", 0.0)
                is_flag = item_row.get("is_potential_discrepancy", False)

                if is_flag:
                    st.error(f"🚩 **FLAGGED FOR REVIEW:** Markup of **{var_pct:+.1f}%** exceeds threshold (+{discrepancy_threshold}%)")
                else:
                    st.success(f"✅ **WITHIN BENCHMARK:** Variance of **{var_pct:+.1f}%** is within acceptable threshold.")

                st.markdown(
                    f"""
                    **Arithmetic Proof:**
                    - $\\text{{Difference (₹)}} = ₹{item_row['billed_line_amount']:,.2f} - ₹{item_row['reference_line_amount']:,.2f} = \\mathbf{{₹{var_inr:+,.2f}}}$
                    - $\\text{{Variance (\\%)}} = \\frac{{{var_inr:+,.2f}}}{{{item_row['reference_line_amount']:,.2f}}} \\times 100 = \\mathbf{{{var_pct:+.1f}\\%}}$
                    """
                )
                if item_row.get("injected_discrepancy", False):
                    st.info("🧪 **Synthetic Ground Truth:** This item was an intentionally injected test discrepancy.")
            else:
                st.warning(f"🚫 **INELIGIBLE FOR COMPARISON:** Reason: `{item_row.get('eligibility_reason')}`")
                st.markdown(
                    """
                    *Note:* Ineligible rows are not evaluated against the review threshold to prevent misleading overcharge claims.
                    """
                )

# =============================================================
# TAB 3: DATA QUALITY & VALIDATION
# =============================================================
with tab_dq:
    st.subheader("Data Quality Governance & Synthetic Test-Set Validation")
    st.caption("Verification of dataset completeness, unit integrity, calculation consistency, and algorithm performance.")

    # DQ Metric Cards
    dq_c1, dq_c2, dq_c3, dq_c4, dq_c5 = st.columns(5)
    with dq_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Duplicate Rows</div>
                <div class="metric-value">{"0" if deduplicate_data else str(dq_report.duplicate_rows_count)}</div>
                <div class="metric-sub">{"Deduplicated" if deduplicate_data else f"{dq_report.duplicate_rows_count} raw duplicates detected"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dq_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Invalid Quantities</div>
                <div class="metric-value">{dq_report.invalid_quantity_count}</div>
                <div class="metric-sub">Non-positive (<=0) or null</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dq_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Math Mismatches</div>
                <div class="metric-value">{dq_report.arithmetic_mismatch_count}</div>
                <div class="metric-sub">Billed != (Qty * Rate)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dq_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Unit Mismatches</div>
                <div class="metric-value">{dq_report.unit_mismatch_count}</div>
                <div class="metric-sub">Pack vs Single unit discrepancies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dq_c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Unmatched / Missing</div>
                <div class="metric-value">{dq_report.unmatched_item_count + dq_report.missing_reference_count}</div>
                <div class="metric-sub">Items lacking catalog match</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Synthetic Test-Set Confusion Matrix & Evaluation
    st.subheader("Synthetic Test-Dataset Validation (Ground Truth Evaluation)")
    st.markdown(
        """
        Because synthetic test records contain known injected discrepancies (`injected_discrepancy = True`),
        we evaluate the algorithmic threshold rule against ground truth labels on comparable records.
        """
    )

    val_metrics = evaluate_test_dataset_performance(processed_df)

    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        st.markdown("##### Performance Metrics")
        st.write(f"**Evaluation Population:** {val_metrics.total_labeled_items:,} comparable line items")

        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        p_c1.metric("Precision", f"{val_metrics.precision:.1f}%")
        p_c2.metric("Recall", f"{val_metrics.recall:.1f}%")
        p_c3.metric("F1-Score", f"{val_metrics.f1_score:.1f}%")
        p_c4.metric("Accuracy", f"{val_metrics.accuracy:.1f}%")

        st.markdown(
            f"""
            <div style="background:#F1F5F9; padding:0.9rem; border-radius:6px; font-size:0.85rem; color:#334155; margin-top:1rem;">
                <b>Metric Definitions:</b><br>
                • <b>Precision:</b> {val_metrics.true_positives} / ({val_metrics.true_positives} + {val_metrics.false_positives}) = True discrepancies / All flagged items.<br>
                • <b>Recall:</b> {val_metrics.true_positives} / ({val_metrics.true_positives} + {val_metrics.false_negatives}) = Captured discrepancies / All true injected anomalies.<br>
                • <b>False Positives ({val_metrics.false_positives}):</b> Normal items whose legitimate institutional pricing exceeded the review threshold (+{discrepancy_threshold}%).<br>
                • <b>False Negatives ({val_metrics.false_negatives}):</b> Injected anomalies whose markup fell below the chosen threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v_col2:
        fig_cm = plot_confusion_matrix_heatmap(
            tp=val_metrics.true_positives,
            fp=val_metrics.false_positives,
            fn=val_metrics.false_negatives,
            tn=val_metrics.true_negatives,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.warning(
        """
        **CRITICAL STATISTICAL LIMITATION:**
        High precision and recall on synthetic benchmark test data demonstrate the internal mechanical consistency of the
        threshold rule against known artificial distributions. **It does not establish real-world clinical accuracy**,
        generalizability across unstandardized hospital chargemasters, or legal admissibility as proven overcharges.
        """
    )

# =============================================================
# TAB 4: BUSINESS INSIGHTS & PRACTICAL OPERATIONAL RECOMMENDATIONS
# =============================================================
with tab_insights:
    st.subheader("Actionable Operational Recommendations")
    st.caption("Data-driven operational priorities derived systematically from calculated coverage, discrepancies, and data defects.")

    # Dynamically generated insights based on calculated metrics
    rec_cards = []

    # 1. Coverage gap recommendation
    min_cov_cat = category_summary.loc[category_summary["Item Coverage (%)"].idxmin()]
    if min_cov_cat["Item Coverage (%)"] < 80.0:
        rec_cards.append(
            {
                "priority": "HIGH",
                "title": f"Expand Reference Benchmark Catalog for {min_cov_cat['Category']}",
                "metric": f"{min_cov_cat['Item Coverage (%)']:.1f}% Item Coverage",
                "action": (
                    f"The **{min_cov_cat['Category']}** category exhibits the lowest benchmark coverage ({min_cov_cat['Item Coverage (%)']:.1f}%). "
                    f"VoltEdge should partner with state tender repositories or specialized formulary catalogs to map missing SKUs "
                    f"before applying automated review rules."
                ),
            }
        )

    # 2. Largest discrepancy category recommendation
    max_disc_cat = category_summary.loc[category_summary["Potential Discrepancy Value (₹)"].idxmax()]
    rec_cards.append(
        {
            "priority": "HIGH",
            "title": f"Prioritize Clinical Human Review in {max_disc_cat['Category']}",
            "metric": f"₹{max_disc_cat['Potential Discrepancy Value (₹)']:,.0f} Flagged Value",
            "action": (
                f"**{max_disc_cat['Category']}** accounts for the largest aggregate potential discrepancy value "
                f"(₹{max_disc_cat['Potential Discrepancy Value (₹)']:,.0f} across {max_disc_cat['Discrepancy Items']} items). "
                f"Operational audit teams should prioritize this domain for itemized clinician spot-checks."
            ),
        }
    )

    # 3. Unit standardization recommendation
    if dq_report.unit_mismatch_count > 0:
        rec_cards.append(
            {
                "priority": "MEDIUM",
                "title": "Deploy Packaging Unit-Conversion Normalizer",
                "metric": f"{dq_report.unit_mismatch_count} Unit Incompatibilities",
                "action": (
                    f"Identified {dq_report.unit_mismatch_count} line items where unit mismatches (e.g. 'pack_of_10' vs 'piece') "
                    f"prevented benchmark comparison. Implementing an automated multiplier dictionary for standard hospital packaging "
                    f"will recover up to {round(dq_report.unit_mismatch_count / len(raw_df) * 100, 1)}% of currently unmatchable rows."
                ),
            }
        )

    # 4. Coverage guardrail recommendation
    if exec_metrics.item_benchmark_coverage_pct < 75.0:
        rec_cards.append(
            {
                "priority": "CRITICAL GUARDRAIL",
                "title": "Restrict Broad Overcharge Claims Under Limited Coverage",
                "metric": f"{exec_metrics.item_benchmark_coverage_pct:.1f}% Overall Coverage",
                "action": (
                    "Total dataset benchmark coverage is below 75%. VoltEdge leadership must refrain from making institutional "
                    "or media claims regarding overall hospital overcharging rates until unmapped items are reconciled."
                ),
            }
        )
    else:
        rec_cards.append(
            {
                "priority": "MAINTENANCE",
                "title": "Continuous Benchmark Refresh Protocol",
                "metric": "Quarterly Schedule",
                "action": (
                    "Establish a 90-day cadence to re-index median reference rates. Medical inflation and wholesale pharmaceutical "
                    "changes can artificially elevate potential discrepancy rates if benchmarks remain static."
                ),
            }
        )

    for rec in rec_cards:
        color = "#DC2626" if "HIGH" in rec["priority"] or "CRITICAL" in rec["priority"] else "#0F766E"
        st.markdown(
            f"""
            <div style="border:1px solid #E2E8F0; border-left:5px solid {color}; padding:1.2rem; border-radius:6px; margin-bottom:1rem; background:#F8FAFC;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h5 style="margin:0; color:#1E293B;">{rec['title']}</h5>
                    <span style="background:{color}20; color:{color}; font-weight:700; font-size:0.75rem; padding:3px 8px; border-radius:4px;">
                        {rec['priority']} · {rec['metric']}
                    </span>
                </div>
                <p style="margin:0.5rem 0 0 0; color:#475569; font-size:0.92rem; line-height:1.5;">
                    {rec['action']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================
# TAB 5: PILOT ECONOMICS SIMULATOR (BUSINESS MODEL)
# =============================================================
with tab_economics:
    st.subheader("SaveOnCare Pilot Unit Economics & Financial Simulator")
    st.caption("Interactive commercial model for institutional enterprise pilots with hospitals, TPAs, and corporate insurers.")

    st.markdown(
        """
        <div style="background:#EFF6FF; border-left:4px solid #3B82F6; padding:0.75rem 1rem; border-radius:4px; font-size:0.88rem; color:#1E40AF; margin-bottom:1rem;">
            <b>Financial Modeling Scope:</b> Illustrative assumptions for operational planning.
            Strictly distinguishes <i>Gross Assumed Revenue</i>, <i>Variable Direct Costs</i>, <i>Fixed Operating Support</i>, and <i>Net Operating Surplus</i>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_col1, sim_col2 = st.columns([1, 1.4])

    with sim_col1:
        st.markdown("##### 🎛️ Pilot Commercial Assumptions")

        p_volume = st.number_input("Number of Bills Processed in Pilot", min_value=100, max_value=20000, value=1500, step=100)
        p_price = st.number_input("Contract Price Charged per Bill (₹)", min_value=50.0, max_value=2000.0, value=350.0, step=25.0)
        p_setup = st.number_input("One-Time Institutional Setup / Integration Fee (₹)", min_value=0.0, max_value=500000.0, value=50000.0, step=10000.0)
        p_var_cost = st.number_input("Variable Cost per Bill (₹) (OCR, parsing, clinical QA)", min_value=20.0, max_value=1000.0, value=120.0, step=10.0)
        p_ops = st.number_input("Monthly Fixed Support & Cloud Costs (₹)", min_value=5000.0, max_value=300000.0, value=25000.0, step=5000.0)
        p_duration = st.slider("Pilot Duration (Months)", min_value=1, max_value=12, value=6)

    # Run simulation
    sim_res = run_pilot_economics_simulation(
        bills_processed=int(p_volume),
        contract_price_per_bill_inr=float(p_price),
        setup_fee_inr=float(p_setup),
        variable_cost_per_bill_inr=float(p_var_cost),
        monthly_support_ops_inr=float(p_ops),
        pilot_duration_months=int(p_duration),
    )

    with sim_col2:
        st.markdown("##### 📊 Financial Outputs & Margins")

        out_c1, out_c2 = st.columns(2)
        with out_c1:
            st.metric("Gross Assumed Revenue", f"₹{sim_res.assumed_revenue_inr:,.0f}")
            st.metric("Total Modelled Costs", f"₹{sim_res.total_modelled_costs_inr:,.0f}")
            st.metric("Cost per Processed Bill", f"₹{sim_res.cost_per_processed_bill_inr:,.2f}")

        with out_c2:
            surplus_color = "normal" if sim_res.net_operating_surplus_inr >= 0 else "inverse"
            st.metric("Net Operating Surplus", f"₹{sim_res.net_operating_surplus_inr:,.0f}", delta=f"{sim_res.contribution_margin_pct:.1f}% Contribution Margin")
            st.metric(
                "Break-Even Volume",
                f"{int(sim_res.break_even_bills):,} bills" if sim_res.break_even_bills is not None else "Negative Unit Margin",
                delta="Achieved in Pilot" if sim_res.break_even_achieved else "Below Break-Even",
            )

        # Break-Even Visualisation
        vol_range = list(range(100, max(int(p_volume) * 2, 2000), 100))
        rev_curve = [(v * p_price + p_setup) for v in vol_range]
        cost_curve = [(v * p_var_cost + p_ops * p_duration) for v in vol_range]

        fig_be = plot_break_even_curve(
            bills_range=vol_range,
            revenues=rev_curve,
            total_costs=cost_curve,
            break_even_point=sim_res.break_even_bills,
        )
        st.plotly_chart(fig_be, use_container_width=True)

    # Multi-scenario sensitivity heatmap
    st.markdown("---")
    st.markdown("##### 🎯 Multi-Scenario Sensitivity Heatmap (Net Operating Surplus in ₹)")
    st.caption("Impact of variations in bill volume (-50% to +50%) and contracted bill fee (-20% to +20%).")

    sens_matrix = generate_sensitivity_matrix(sim_res)
    fig_sens = plot_sensitivity_heatmap(sens_matrix)
    st.plotly_chart(fig_sens, use_container_width=True)

    # Assumption Elasticity Ranking
    st.markdown("##### ⚖️ Assumption Elasticity & Impact Hierarchy")
    st.caption("Ranked by percentage impact on Net Operating Surplus resulting from a +10% isolated shift in each parameter.")
    st.dataframe(pd.DataFrame(sim_res.elasticity_ranking), use_container_width=True, hide_index=True)

# =============================================================
# TAB 6: EXPORT & AUDIT TRAIL
# =============================================================
with tab_audit:
    st.subheader("Data Export, Audit Trail & Regulatory Readiness")
    st.caption("Download standardized outputs for external audits, compliance reviews, and executive reporting.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("##### 📥 Export Filtered Analysis Dataset")
        st.write(f"Contains **{len(processed_df):,}** processed line items with computed variance amounts and audit flags.")

        csv_buffer = StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ Download Processed Dataset (CSV)",
            data=csv_bytes,
            file_name=f"saveoncare_analysis_{datetime.date.today().isoformat()}.csv",
            mime="text/csv",
        )

    with exp_col2:
        st.markdown("##### 📄 Export Data Quality & Audit Report")
        st.write("Comprehensive summary of schema validation, integrity checks, and parameter assumptions.")

        report_md = f"""# SaveOnCare Intelligence Lab — Audit & Compliance Report
**Generated Date:** {datetime.datetime.now().isoformat()}
**Platform:** VoltEdge Healthcare Intelligence
**Dataset Records Analyzed:** {len(processed_df):,}
**Active Review Threshold:** +{discrepancy_threshold}%

## Executive Summary
- Total Billed Amount: INR {exec_metrics.total_billed_amount_inr:,.2f}
- Benchmark Item Coverage: {exec_metrics.item_benchmark_coverage_pct:.1f}%
- Benchmark Value Coverage: {exec_metrics.value_benchmark_coverage_pct:.1f}%
- Potential Discrepancy Items: {exec_metrics.potential_discrepancy_count:,} ({exec_metrics.potential_discrepancy_rate_pct:.1f}%)
- Potential Discrepancy Value: INR {exec_metrics.potential_discrepancy_value_inr:,.2f}

## Data Quality Summary
- Deduplication Applied: {deduplicate_data}
- Missing Value Rate: {dq_report.missing_value_rate:.2f}%
- Invalid Quantities (<=0): {dq_report.invalid_quantity_count}
- Arithmetic Mismatches: {dq_report.arithmetic_mismatch_count}
- Unit Mismatches: {dq_report.unit_mismatch_count}
- Non-Positive Reference Prices: {dq_report.non_positive_reference_count}

## Real-World Deployment Checklist
1. Clinical formulary verification with hospital pharmacy therapeutics committee.
2. Direct integration with official state tender schedules.
3. Multi-tier hospital stratification accounting for NABH accreditation status.
4. Independent legal verification distinguishing clinical pricing variance from statutory overcharges.
"""
        st.download_button(
            label="⬇️ Download Audit Report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name=f"saveoncare_audit_report_{datetime.date.today().isoformat()}.md",
            mime="text/markdown",
        )

    st.markdown("---")
    st.subheader("🚦 Roadmap & Real-World Deployment Prerequisite Evidence")
    st.markdown(
        """
        Before transitioning this analytical prototype from synthetic data to live hospital billing audits,
        VoltEdge must establish the following statutory and technical requirements:

        1. **Verified Clinical Ontology Mapping:** Transition from string description matching to standardized SNOMED CT and LOINC codes.
        2. **Statutory Benchmark Sourcing:** Replace synthetic medians with verified NPPA ceiling prices, CGHS rate schedules, and bilateral insurer PPO contracts.
        3. **Hospital Tier Stratification:** Distinguish tertiary academic medical centers from secondary nursing homes (accreditation, NABH, JCI).
        4. **Clinical Nuance Exception Engine:** Account for surgical complications, patient comorbidities, and emergent unbundling rules.
        5. **Strict Legal Governance:** Ensure all platform outputs explicitly cite reference source timestamps and disclaim judicial overcharge claims without independent medical board review.
        """
    )
