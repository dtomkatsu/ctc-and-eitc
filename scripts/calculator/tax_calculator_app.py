#!/usr/bin/env python3
"""
Hawaii State Income Tax Calculator

Compares three scenarios side by side:
  - Act 46 (2025 baseline): current law
  - HB 2306 HD1: top 3 rates +1pp, enhanced refundable CDCC ($10k/$20k caps, 50%→5%)
  - SB 3125 SD1: expanded bracket structure, enhanced refundable CDCC ($10k/$20k, formula*)

* SB 3125 CDCC applicable percentage has blanks in the enrolled bill.
  This calculator assumes 35% starting at $43,000 AGI, reduced 1pp per $3,000, floor 15%.
"""

import sys
from pathlib import Path
import math

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tax.config import TaxSystemConfig, TaxSystemRegistry, TaxCalculator
from src.tax.adjustments.hawaii_credits import HawaiiTaxCredits


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hawaii Tax Calculator",
    page_icon="🌺",
    layout="wide",
)

st.title("🌺 Hawaii State Income Tax Calculator")
st.caption(
    "Compares Act 46 (current law) vs HB 2306 HD1 vs SB 3125 SD1 — Tax Year 2025/2027"
)

# ── Scenarios ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_calculator():
    return TaxCalculator(PROJECT_ROOT)

@st.cache_resource
def get_scenarios():
    return {
        "Act 46 (2025 Law)": {
            "config": TaxSystemRegistry.get_act46_2025_system(),
            "credit_scenario": None,
            "year": 2025,
            "color": "#1f77b4",
        },
        "HB 2306 HD1": {
            "config": TaxSystemRegistry.get_hb2306_hd1_2027_system(),
            "credit_scenario": "hb2306_hd1",
            "year": 2027,
            "color": "#d62728",
        },
        "SB 3125 SD1": {
            "config": TaxSystemRegistry.get_sb3125_sd1_2027_system(),
            "credit_scenario": "sb3125_sd1",
            "year": 2027,
            "color": "#2ca02c",
        },
    }

calculator = get_calculator()
scenarios = get_scenarios()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Information")

    income = st.number_input(
        "Gross Income (AGI)",
        min_value=0,
        max_value=5_000_000,
        value=75_000,
        step=1_000,
        format="%d",
        help="Adjusted Gross Income",
    )

    filing_status = st.selectbox(
        "Filing Status",
        options=[
            "single",
            "married_filing_jointly",
            "head_of_household",
            "married_filing_separately",
        ],
        format_func=lambda s: {
            "single": "Single",
            "married_filing_jointly": "Married Filing Jointly",
            "head_of_household": "Head of Household",
            "married_filing_separately": "Married Filing Separately",
        }[s],
    )

    num_dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=10, value=0, step=1
    )

    st.divider()
    st.subheader("Childcare Expenses")
    use_actual_expenses = st.checkbox(
        "Enter actual childcare expenses",
        value=False,
        disabled=(num_dependents == 0),
    )
    actual_expenses = None
    if use_actual_expenses and num_dependents > 0:
        actual_expenses = st.number_input(
            "Annual childcare expenses ($)",
            min_value=0,
            max_value=50_000,
            value=min(12_000 * num_dependents, 20_000),
            step=500,
        )

    st.divider()
    st.caption(
        "**Note:** SB 3125 CDCC applicable percentage uses assumed values — "
        "the enacted bill has blanks. See footnote in results."
    )


# ── Calculation helper ────────────────────────────────────────────────────────
def compute(scenario_name: str, agi: float, fs: str, deps: int, exp_override=None):
    s = scenarios[scenario_name]
    config = s["config"]
    credit_scenario = s["credit_scenario"]
    year = s["year"]

    num_exemptions = (2 if fs == "married_filing_jointly" else 1) + deps

    result = calculator.calculate_tax(agi, config, fs, num_exemptions)
    tax = result["tax_liability"]

    credit_calc = HawaiiTaxCredits(year)

    # Override childcare expense if user supplied one
    if exp_override is not None and deps > 0:
        # Temporarily monkey-patch expense estimate — compute CDCC manually
        cdcc = _cdcc_with_expenses(credit_calc, agi, fs, deps, credit_scenario, exp_override)
        credits_raw = credit_calc.calculate_total_credits(agi, fs, deps, tax, credit_scenario)
        # Replace child_care with overridden value
        old_cc = credits_raw["child_care"]
        credits_raw["child_care"] = cdcc

        enhanced_cdcc = credit_scenario in ("hb2306_hd1", "sb3125_sd1")
        refundable = credits_raw["food_excise"] + credits_raw["renters"]
        nonrefundable = credits_raw["renewable_energy"]
        if enhanced_cdcc:
            refundable += cdcc
        else:
            nonrefundable = min(nonrefundable + cdcc, tax)

        credits_raw["total_refundable"] = refundable
        credits_raw["total_nonrefundable"] = nonrefundable
        credits_raw["total"] = refundable + nonrefundable
        credits = credits_raw
    else:
        credits = credit_calc.calculate_total_credits(agi, fs, deps, tax, credit_scenario)

    net_tax = tax - credits["total"]
    effective_rate = (net_tax / agi * 100) if agi > 0 else 0

    return {
        **result,
        "credits": credits,
        "net_tax": net_tax,
        "net_effective_rate": effective_rate,
    }


def _cdcc_with_expenses(credit_calc, agi, fs, deps, credit_scenario, expenses):
    """Compute CDCC using caller-supplied expense amount instead of the Hawaii estimate."""
    if deps == 0:
        return 0.0

    if credit_scenario == "hb2306_hd1":
        expense_cap = 10_000 if deps == 1 else 20_000
        if agi <= 80_000:
            pct = 0.50
        elif agi <= 90_000:
            pct = 0.45
        elif agi <= 100_000:
            pct = 0.40
        elif agi <= 110_000:
            pct = 0.35
        elif agi <= 120_000:
            pct = 0.30
        elif agi <= 130_000:
            pct = 0.25
        elif agi <= 140_000:
            pct = 0.20
        elif agi <= 150_000:
            pct = 0.15
        elif agi <= 160_000:
            pct = 0.10
        else:
            pct = 0.05
    elif credit_scenario == "sb3125_sd1":
        expense_cap = 10_000 if deps == 1 else 20_000
        threshold = 43_000
        steps = math.ceil(max(0, agi - threshold) / 3_000)
        pct = max(0.15, 0.35 - 0.01 * steps)
    else:
        expense_cap = 3_000 if deps == 1 else 6_000
        if agi > 100_000:
            return 0.0
        if agi <= 25_000:
            pct = 0.25
        elif agi <= 43_000:
            pct = 0.25 - 0.05 * (agi - 25_000) / 18_000
        else:
            pct = 0.20

    return min(expenses, expense_cap) * pct


# ── Compute all scenarios ─────────────────────────────────────────────────────
results = {
    name: compute(name, income, filing_status, num_dependents, actual_expenses)
    for name in scenarios
}

baseline_net = results["Act 46 (2025 Law)"]["net_tax"]


# ── Section 1: Summary cards ──────────────────────────────────────────────────
st.header("Summary")
cols = st.columns(3)
for col, (name, r) in zip(cols, results.items()):
    color = scenarios[name]["color"]
    diff = r["net_tax"] - baseline_net
    diff_str = f"{diff:+,.0f}" if name != "Act 46 (2025 Law)" else ""
    diff_label = f"vs Act 46: **${diff_str}**" if diff_str else "Baseline"

    col.markdown(
        f"""
        <div style="border:1px solid {color}; border-radius:8px; padding:16px;">
            <h4 style="color:{color}; margin:0 0 8px 0">{name}</h4>
            <p style="font-size:1.8rem; font-weight:700; margin:0">${r['net_tax']:,.0f}</p>
            <p style="margin:4px 0 0 0; color:#666">Net Tax</p>
            <p style="margin:4px 0 0 0; font-size:0.85rem">{diff_label}</p>
            <p style="margin:4px 0 0 0; font-size:0.85rem">
                Effective rate: {r['net_effective_rate']:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Section 2: Detailed comparison table ─────────────────────────────────────
st.header("Detailed Comparison")

scenario_names = list(results.keys())
r0, r1, r2 = [results[n] for n in scenario_names]

table_rows = [
    ("Gross Income (AGI)", income, income, income),
    ("Standard Deduction", r0["standard_deduction"], r1["standard_deduction"], r2["standard_deduction"]),
    ("Personal Exemptions", r0["personal_exemptions"], r1["personal_exemptions"], r2["personal_exemptions"]),
    ("Taxable Income", r0["taxable_income"], r1["taxable_income"], r2["taxable_income"]),
    ("Tax Before Credits", r0["tax_liability"], r1["tax_liability"], r2["tax_liability"]),
    ("— Food/Excise Credit", r0["credits"]["food_excise"], r1["credits"]["food_excise"], r2["credits"]["food_excise"]),
    ("— CDCC", r0["credits"]["child_care"], r1["credits"]["child_care"], r2["credits"]["child_care"]),
    ("— Renters Credit", r0["credits"]["renters"], r1["credits"]["renters"], r2["credits"]["renters"]),
    ("Total Credits", r0["credits"]["total"], r1["credits"]["total"], r2["credits"]["total"]),
    ("Net Tax", r0["net_tax"], r1["net_tax"], r2["net_tax"]),
]

def fmt(v):
    if v < 0:
        return f"(${abs(v):,.0f})"
    return f"${v:,.0f}"

table_data = {
    "": [row[0] for row in table_rows],
    scenario_names[0]: [fmt(row[1]) for row in table_rows],
    scenario_names[1]: [fmt(row[2]) for row in table_rows],
    scenario_names[2]: [fmt(row[3]) for row in table_rows],
}

bold_rows = {"Tax Before Credits", "Net Tax", "Total Credits", "Taxable Income"}
df_table = pd.DataFrame(table_data)
st.dataframe(df_table, use_container_width=True, hide_index=True)

rate_rows = [
    ("Marginal Rate", f"{r0['marginal_rate']:.1f}%", f"{r1['marginal_rate']:.1f}%", f"{r2['marginal_rate']:.1f}%"),
    ("Effective Rate (pre-credits)", f"{r0['effective_rate']:.2f}%", f"{r1['effective_rate']:.2f}%", f"{r2['effective_rate']:.2f}%"),
    ("Net Effective Rate", f"{r0['net_effective_rate']:.2f}%", f"{r1['net_effective_rate']:.2f}%", f"{r2['net_effective_rate']:.2f}%"),
    ("vs Act 46 (net tax)", "—",
     f"${r1['net_tax']-baseline_net:+,.0f}",
     f"${r2['net_tax']-baseline_net:+,.0f}"),
]
df_rates = pd.DataFrame(
    {"": [r[0] for r in rate_rows],
     scenario_names[0]: [r[1] for r in rate_rows],
     scenario_names[1]: [r[2] for r in rate_rows],
     scenario_names[2]: [r[3] for r in rate_rows]}
)
st.dataframe(df_rates, use_container_width=True, hide_index=True)


# ── Section 3: Income sweep chart ────────────────────────────────────────────
st.header("Net Tax vs Income")

@st.cache_data
def build_sweep(fs, deps, exp_override=None):
    incomes = list(range(0, 100_001, 2_000)) + list(range(100_000, 500_001, 5_000))
    sweep = {name: [] for name in scenarios}
    for inc in incomes:
        for name in scenarios:
            r = compute(name, inc, fs, deps, exp_override)
            sweep[name].append(r["net_tax"])
    return incomes, sweep

incomes, sweep = build_sweep(filing_status, num_dependents, actual_expenses)

fig_sweep = go.Figure()
for name, color_info in scenarios.items():
    fig_sweep.add_trace(go.Scatter(
        x=incomes,
        y=sweep[name],
        name=name,
        line=dict(color=color_info["color"], width=2),
        hovertemplate="%{x:$,.0f} income → %{y:$,.0f} net tax<extra>" + name + "</extra>",
    ))
# Mark current income
fig_sweep.add_vline(x=income, line_dash="dot", line_color="gray", opacity=0.6,
                    annotation_text=f"${income:,.0f}", annotation_position="top right")
fig_sweep.add_hline(y=0, line_color="black", line_width=0.5)
fig_sweep.update_layout(
    xaxis_title="Gross Income (AGI)",
    yaxis_title="Net Tax ($)",
    xaxis_tickformat="$,.0f",
    yaxis_tickformat="$,.0f",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=420,
    margin=dict(t=40),
)
st.plotly_chart(fig_sweep, use_container_width=True)


# ── Section 4: Tax breakdown chart ───────────────────────────────────────────
st.header("Tax Breakdown by Scenario")

fig_bar = go.Figure()
credit_labels = ["Food/Excise Credit", "CDCC", "Renters Credit"]
credit_keys = ["food_excise", "child_care", "renters"]
credit_colors = ["#aec7e8", "#98df8a", "#c5b0d5"]

for name, r in results.items():
    color = scenarios[name]["color"]
    fig_bar.add_trace(go.Bar(
        name=f"{name} — Tax Before Credits",
        x=[name],
        y=[r["tax_liability"]],
        marker_color=color,
        opacity=0.85,
        hovertemplate=f"Tax before credits: $%{{y:,.0f}}<extra>{name}</extra>",
    ))

for label, key, cc in zip(credit_labels, credit_keys, credit_colors):
    for name, r in results.items():
        fig_bar.add_trace(go.Bar(
            name=label,
            x=[name],
            y=[-r["credits"][key]],
            marker_color=cc,
            hovertemplate=f"{label}: $%{{y:,.0f}}<extra>{name}</extra>",
            showlegend=(name == "Act 46 (2025 Law)"),
        ))

fig_bar.update_layout(
    barmode="relative",
    yaxis_title="Amount ($)",
    yaxis_tickformat="$,.0f",
    height=380,
    margin=dict(t=30),
)
st.plotly_chart(fig_bar, use_container_width=True)


# ── Section 5: CDCC detail (only if dependents > 0) ──────────────────────────
if num_dependents > 0:
    st.header("CDCC Detail")
    st.caption(
        "Child and Dependent Care Credit — Hawaii HRS §235-55.6. "
        "Childcare expenses estimated at $12,000/child/year (Hawaii average) unless you entered actual expenses above."
    )

    def cdcc_pct_label(name, agi, deps):
        cs = scenarios[name]["credit_scenario"]
        if cs == "hb2306_hd1":
            if agi <= 80_000: pct = 50
            elif agi <= 90_000: pct = 45
            elif agi <= 100_000: pct = 40
            elif agi <= 110_000: pct = 35
            elif agi <= 120_000: pct = 30
            elif agi <= 130_000: pct = 25
            elif agi <= 140_000: pct = 20
            elif agi <= 150_000: pct = 15
            elif agi <= 160_000: pct = 10
            else: pct = 5
            return f"{pct}%"
        elif cs == "sb3125_sd1":
            steps = math.ceil(max(0, agi - 43_000) / 3_000)
            pct = max(15, 35 - steps)
            return f"{pct}% (assumed*)"
        else:
            if agi > 100_000: return "0% (phased out)"
            if agi <= 25_000: pct = 25
            elif agi <= 43_000:
                pct = round(25 - 5 * (agi - 25_000) / 18_000, 1)
            else: pct = 20
            return f"{pct}%"

    def expense_cap_label(name, deps):
        cs = scenarios[name]["credit_scenario"]
        if cs in ("hb2306_hd1", "sb3125_sd1"):
            cap = 10_000 if deps == 1 else 20_000
        else:
            cap = 3_000 if deps == 1 else 6_000
        return f"${cap:,.0f}"

    exp_used = actual_expenses if actual_expenses is not None else min(12_000 * num_dependents, 20_000)

    cdcc_rows = []
    for name, r in results.items():
        cs = scenarios[name]["credit_scenario"]
        refundable = cs in ("hb2306_hd1", "sb3125_sd1")
        cdcc_rows.append({
            "Scenario": name,
            "Applicable %": cdcc_pct_label(name, income, num_dependents),
            "Expense Cap": expense_cap_label(name, num_dependents),
            "Estimated Expenses": f"${exp_used:,.0f}",
            "CDCC Amount": f"${r['credits']['child_care']:,.0f}",
            "Refundable?": "Yes" if refundable else "No",
        })

    st.dataframe(pd.DataFrame(cdcc_rows), use_container_width=True, hide_index=True)

    st.caption(
        "\\* **SB 3125 SD1 assumption**: The enacted bill has blank values for the starting percentage and "
        "AGI threshold in §235-55.6. This calculator uses **35% starting at $43,000 AGI**, reduced by "
        "1 percentage point per $3,000 of AGI above that threshold, with a floor of **15%**. "
        "Actual enacted values may differ."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Assumptions**: (1) All income is wages/earned income. "
    "(2) Standard deduction taken (no itemizing). "
    "(3) Childcare expenses estimated at $12,000/child/year unless overridden. "
    "(4) Act 46 modeled at TY2025; HB 2306 HD1 and SB 3125 SD1 modeled at TY2027. "
    "(5) Renewable energy credit not modeled (highly variable; affects ~2% of filers). "
    "(6) SB 3125 Part II business credit repeals (effective 1/1/2029) not modeled in individual calculator."
)
