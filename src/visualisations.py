"""
Interactive Plotly Visualisations for SaveOnCare Intelligence Lab.

Provides professional BI charts: category breakdowns, variance distributions,
monthly time trends, confusion matrices, coverage gauges, and pilot economics curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Theme Color Palette
COLORS = {
    "primary": "#0F766E",     # Deep Teal
    "primary_light": "#14B8A6",
    "secondary": "#1E293B",   # Slate Navy
    "discrepancy": "#E11D48", # Vibrant Rose / Warning
    "benchmark": "#2563EB",   # Benchmark Blue
    "success": "#059669",     # Emerald
    "muted": "#94A3B8",       # Slate 400
    "bg_card": "#F8FAFC",
}


def plot_category_billed_vs_discrepancy(category_df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side bar chart showing Total Billed vs Potential Discrepancy Value by Category.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=category_df["Category"],
            y=category_df["Total Billed (₹)"],
            name="Total Billed (₹)",
            marker_color="#0F766E",
            hovertemplate="<b>%{x}</b><br>Total Billed: ₹%{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=category_df["Category"],
            y=category_df["Potential Discrepancy Value (₹)"],
            name="Potential Discrepancy (₹)",
            marker_color="#E11D48",
            hovertemplate="<b>%{x}</b><br>Potential Discrepancy: ₹%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Billed Amount vs. Potential Discrepancy by Category</b>",
        barmode="group",
        xaxis_title="Category",
        yaxis_title="Amount in INR (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_variance_distribution(
    processed_df: pd.DataFrame,
    threshold_pct: float = 20.0,
) -> go.Figure:
    """
    Histogram and KDE-like distribution of percentage variances for comparable items,
    highlighting the user-defined threshold boundary.
    """
    comp_df = processed_df[processed_df["is_comparable"] == True].copy()
    if comp_df.empty or "variance_pct" not in comp_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No comparable data points to plot distribution", showarrow=False)
        return fig

    # Clip extreme outliers for visual clarity on histogram (e.g., -50% to +250%)
    variances = comp_df["variance_pct"].clip(-50, 250)

    fig = px.histogram(
        variances,
        x="variance_pct",
        nbins=40,
        title=f"<b>Distribution of Percentage Variances (Benchmark Line: +{threshold_pct:.0f}%)</b>",
        labels={"variance_pct": "Percentage Variance from Benchmark (%)"},
        color_discrete_sequence=["#0F766E"],
        opacity=0.85,
    )

    # Add vertical line for threshold
    fig.add_vline(
        x=threshold_pct,
        line_width=2.5,
        line_dash="dash",
        line_color="#E11D48",
        annotation_text=f"Review Threshold ({threshold_pct:+.0f}%)",
        annotation_position="top right",
    )

    fig.add_vline(
        x=0.0,
        line_width=1.5,
        line_dash="dot",
        line_color="#64748B",
        annotation_text="Parity (0%)",
        annotation_position="top left",
    )

    fig.update_layout(
        yaxis_title="Line Item Count",
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def plot_monthly_trend(processed_df: pd.DataFrame) -> go.Figure:
    """
    Monthly time series chart showing billed amount and potential discrepancy volume over 2024.
    """
    df_trend = processed_df.copy()
    if "bill_date" not in df_trend.columns or df_trend.empty:
        fig = go.Figure()
        return fig

    df_trend["bill_date"] = pd.to_datetime(df_trend["bill_date"])
    df_trend["month"] = df_trend["bill_date"].dt.to_period("M").astype(str)

    monthly_summary = (
        df_trend.groupby("month")
        .agg(
            total_billed=("billed_line_amount", "sum"),
            potential_discrepancy=("potential_discrepancy_value_inr", "sum"),
            line_count=("line_item_id", "count"),
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=monthly_summary["month"],
            y=monthly_summary["total_billed"],
            name="Monthly Billed (₹)",
            marker_color="#0F766E",
            opacity=0.75,
            hovertemplate="<b>%{x}</b><br>Total Billed: ₹%{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_summary["month"],
            y=monthly_summary["potential_discrepancy"],
            name="Potential Discrepancy (₹)",
            line=dict(color="#E11D48", width=3),
            mode="lines+markers",
            hovertemplate="<b>%{x}</b><br>Discrepancy: ₹%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Monthly Billing Volume & Potential Discrepancy Trajectory (2024)</b>",
        xaxis_title="Billing Month",
        yaxis_title="Amount in INR (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_benchmark_coverage_breakdown(category_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked bar chart showing Item Coverage % vs Unmatched/Ineligible % by category.
    """
    cov_df = category_df.copy()
    cov_df["Ineligible (%)"] = 100.0 - cov_df["Item Coverage (%)"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=cov_df["Category"],
            x=cov_df["Item Coverage (%)"],
            name="Comparable Benchmark (%)",
            orientation="h",
            marker_color="#0F766E",
            hovertemplate="<b>%{y}</b><br>Comparable Coverage: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=cov_df["Category"],
            x=cov_df["Ineligible (%)"],
            name="Unmatched / Incompatible (%)",
            orientation="h",
            marker_color="#94A3B8",
            hovertemplate="<b>%{y}</b><br>Unmatched / Ineligible: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Benchmark Coverage Completeness by Category</b>",
        barmode="stack",
        xaxis_title="Share of Line Items (%)",
        xaxis=dict(range=[0, 100]),
        yaxis_title="Category",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def plot_confusion_matrix_heatmap(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> go.Figure:
    """
    Heatmap rendering of the 2x2 confusion matrix against synthetic test ground truth.
    """
    z_values = [[tp, fp], [fn, tn]]
    x_labels = ["Flagged: Discrepancy", "Flagged: Benchmark Range"]
    y_labels = ["Ground Truth: Discrepancy", "Ground Truth: Normal"]

    annotations = []
    cell_descriptions = [
        [f"True Positive (TP)<br><b>{tp}</b>", f"False Positive (FP)<br><b>{fp}</b>"],
        [f"False Negative (FN)<br><b>{fn}</b>", f"True Negative (TN)<br><b>{tn}</b>"],
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=cell_descriptions[i][j],
                    font=dict(size=14, color="white" if z_values[i][j] > max(fp, fn) else "#1E293B"),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#E2E8F0"], [0.2, "#CBD5E1"], [1.0, "#0F766E"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title="<b>Synthetic Test-Set Confusion Matrix (Injected Ground Truth)</b>",
        annotations=annotations,
        xaxis_title="Algorithmic Flagging Rule",
        yaxis_title="Synthetic Benchmark Truth",
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def plot_break_even_curve(
    bills_range: list[int],
    revenues: list[float],
    total_costs: list[float],
    break_even_point: float | None,
) -> go.Figure:
    """
    Pilot economics break-even curve displaying Assumed Revenue vs Total Modelled Costs.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=bills_range,
            y=revenues,
            name="Assumed Revenue (₹)",
            line=dict(color="#0F766E", width=3),
            hovertemplate="Volume: %{x:,} bills<br>Revenue: ₹%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bills_range,
            y=total_costs,
            name="Total Modelled Costs (₹)",
            line=dict(color="#E11D48", width=3, dash="dot"),
            hovertemplate="Volume: %{x:,} bills<br>Total Cost: ₹%{y:,.0f}<extra></extra>",
        )
    )

    if break_even_point is not None and min(bills_range) <= break_even_point <= max(bills_range):
        fig.add_vline(
            x=break_even_point,
            line_width=2,
            line_dash="dash",
            line_color="#2563EB",
            annotation_text=f"Break-Even: {int(break_even_point):,} bills",
            annotation_position="top left",
        )

    fig.update_layout(
        title="<b>Pilot Revenue & Total Cost Trajectory across Bill Volumes</b>",
        xaxis_title="Number of Hospital Bills Processed",
        yaxis_title="Financial Amount (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_sensitivity_heatmap(sensitivity_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap displaying Net Operating Surplus across Volume and Contract Price variations.
    """
    z_matrix = sensitivity_df.values
    x_labels = list(sensitivity_df.columns)
    y_labels = list(sensitivity_df.index)

    text_matrix = [[f"₹{val:,.0f}" for val in row] for row in z_matrix]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=x_labels,
            y=y_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="Tealgrn",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="<b>Sensitivity Matrix: Net Operating Surplus (₹) vs Volume & Fee</b>",
        xaxis_title="Contract Price per Bill Fee",
        yaxis_title="Pilot Bill Processing Volume",
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig
