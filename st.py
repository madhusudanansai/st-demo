# Claims Executive Dashboard v2.5 — C‑suite polish + filters + drilldowns
# ---------------------------------------------------------------
# Streamlit app that generates mock claims data, provides executive KPIs,
# interactive filters, claim-level timeline, financial waterfall, and
# root‑cause analysis. Compatible with Plotly <=5.13 (no text_auto).

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------
# App / Theme Setup
# -----------------------------
st.set_page_config(
    page_title="Claims Executive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, executive CSS polish
st.markdown(
    """
    <style>
    .title { font-size: 28px; font-weight: 800; color: #0f172a; }
    .subtitle { font-size: 15px; color: #475569; }
    .metric-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 18px; }
    .metric-val { font-size: 26px; font-weight:700; color:#0f172a; }
    .metric-lbl { font-size: 12px; color:#64748b; letter-spacing: .3px; }
    .callout { border-left: 4px solid #2563eb; background:#f8fafc; padding:10px 14px; border-radius:6px; }
    .sep { height: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Mock Data Generation
# -----------------------------

def generate_mock_data(n_claims: int = 250, seed: int = 42):
    np.random.seed(seed)

    customers = [
        'EUROGROUP DEUTSCHLAND GMBH', 'Dole Exotics', 'Westfalia',
        'AVOR', 'Prodivec', 'TFC', 'SIIM'
    ]
    plants = ['Guayacán 08', 'EL DURAZNO', 'Banamar 1', 'SAN RAFAEL', 'Palocon', 'Distrito 2', 'Empacadora A', 'Empacadora B']
    reasons_master = ['Scarring', 'Ripe & Turning', 'Crown Rot', 'Neck Rot', 'Latex', 'Peel Rot', 'Mold']
    milestone_names = ['Claim Submission', 'Investigation & Report', 'Claim Approval', 'Credit Note Issued', 'Flow Completion']

    claims = []
    milestones = []

    # Case Study 1 — Stuck
    claims.append({
        'claim_id': 'P6W23BEG36-2025',
        'customer': 'EUROGROUP DEUTSCHLAND GMBH',
        'plant': 'Banamar 1',
        'status': 'Stuck',
        'estimated_amount': 2006.23,
        'final_paid_amount': np.nan,
        'start_date': pd.to_datetime('2025-07-08'),
        'completion_date': pd.NaT,
        'reason': ['Scarring', 'Ripe & Turning', 'Crown Rot', 'Neck Rot', 'Latex', 'Peel Rot']
    })
    milestones.extend([
        {'claim_id': 'P6W23BEG36-2025', 'milestone': 'Claim Submission', 'date': pd.to_datetime('2025-07-08')},
        {'claim_id': 'P6W23BEG36-2025', 'milestone': 'Investigation & Report', 'date': pd.to_datetime('2025-07-29')},
        {'claim_id': 'P6W23BEG36-2025', 'milestone': 'Claim Approval', 'date': pd.NaT},
        {'claim_id': 'P6W23BEG36-2025', 'milestone': 'Credit Note Issued', 'date': pd.NaT},
    ])

    # Case Study 2 — Completed
    claims.append({
        'claim_id': 'P4W13BEG25-2025',
        'customer': 'EUROGROUP DEUTSCHLAND GMBH',
        'plant': 'Guayacán 08',
        'status': 'Completed',
        'estimated_amount': 1742.69,
        'final_paid_amount': 1742.69,
        'start_date': pd.to_datetime('2025-05-31'),
        'completion_date': pd.to_datetime('2025-06-28'),
        'reason': ['Scarring', 'Ripe & Turning', 'Crown Rot', 'Latex', 'Peel Rot']
    })
    milestones.extend([
        {'claim_id': 'P4W13BEG25-2025', 'milestone': 'Claim Submission', 'date': pd.to_datetime('2025-05-31')},
        {'claim_id': 'P4W13BEG25-2025', 'milestone': 'Investigation & Report', 'date': pd.to_datetime('2025-06-09')},
        {'claim_id': 'P4W13BEG25-2025', 'milestone': 'Claim Approval', 'date': pd.to_datetime('2025-06-23')},
        {'claim_id': 'P4W13BEG25-2025', 'milestone': 'Credit Note Issued', 'date': pd.to_datetime('2025-06-27')},
        {'claim_id': 'P4W13BEG25-2025', 'milestone': 'Flow Completion', 'date': pd.to_datetime('2025-06-28')},
    ])

    # Generate others
    for i in range(n_claims - 2):
        claim_id = f'CLM-{i+2:04d}'
        customer = np.random.choice(customers, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.2])
        plant = np.random.choice(plants, p=[0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
        status = np.random.choice(['Completed', 'Cancelled', 'In Progress', 'Stuck'], p=[0.6, 0.1, 0.2, 0.1])
        estimated_amount = round(float(np.random.uniform(500, 5000)), 2)
        start_date = pd.Timestamp(datetime.now() - timedelta(days=int(np.random.randint(1, 240))))

        final_paid_amount = np.nan
        completion_date = pd.NaT
        if status == 'Completed':
            final_paid_amount = round(estimated_amount * float(np.random.uniform(0.7, 1.0)), 2)
            completion_date = start_date + timedelta(days=int(np.random.randint(10, 60)))
        elif status == 'Cancelled':
            final_paid_amount = 0.0

        reasons = list(np.random.choice(reasons_master, size=int(np.random.randint(1, 4)), replace=False))

        claims.append({
            'claim_id': claim_id,
            'customer': customer,
            'plant': plant,
            'status': status,
            'estimated_amount': estimated_amount,
            'final_paid_amount': final_paid_amount,
            'start_date': start_date,
            'completion_date': completion_date,
            'reason': reasons
        })

        # Milestones — approximate spacing
        # Create dates in ascending order, some may be missing for non-completed
        base_dates = sorted([start_date + timedelta(days=int(d)) for d in np.random.randint(4, 45, len(milestone_names))])
        if status == 'Completed':
            n_ms = len(milestone_names)
        else:
            n_ms = int(np.random.randint(1, len(milestone_names)))
        for idx in range(n_ms):
            if idx < len(base_dates):
                milestones.append({
                    'claim_id': claim_id,
                    'milestone': milestone_names[idx],
                    'date': base_dates[idx]
                })

    df_claims = pd.DataFrame(claims)
    df_milestones = pd.DataFrame(milestones)

    # Derivations
    df_claims['days_open'] = (
        np.where(
            df_claims['completion_date'].notna(),
            (df_claims['completion_date'] - df_claims['start_date']).dt.days,
            (pd.Timestamp(datetime.now()) - df_claims['start_date']).dt.days,
        )
    )
    df_claims['cycle_time_days'] = (
        (df_claims['completion_date'] - df_claims['start_date']).dt.days
    )

    return df_claims, df_milestones, reasons_master

# Cache data generation for speed
@st.cache_data(show_spinner=False)
def load_data():
    return generate_mock_data()

# Load
df_claims, df_milestones, reasons_master = load_data()

# -----------------------------
# Helpers
# -----------------------------

def fmt_money(x):
    try:
        return f"${x:,.0f}" if pd.notna(x) else "—"
    except Exception:
        return "—"

def kpi_card(label, value):
    col = st.container()
    with col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-val'>{value}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-lbl'>{label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar — Filters
# -----------------------------
with st.sidebar:
    st.markdown("### Filters")

    # Date range
    min_date = pd.to_datetime(df_claims['start_date']).min().date()
    max_date = pd.to_datetime(df_claims['start_date']).max().date()
    date_range = st.date_input(
        "Start date range",
        value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )

    customers = sorted(df_claims['customer'].dropna().unique().tolist())
    plants = sorted(df_claims['plant'].dropna().unique().tolist())
    statuses = ['In Progress', 'Stuck', 'Completed', 'Cancelled']

    sel_customers = st.multiselect("Customer", customers, default=customers)
    sel_plants = st.multiselect("Plant", plants, default=plants)
    sel_status = st.multiselect("Status", statuses, default=statuses)

    st.markdown("---")
    st.markdown("**Claim selection**")
    # Narrow claim list by filter for the selector
    mask = (
        (df_claims['start_date'].dt.date.between(date_range[0], date_range[1])) &
        (df_claims['customer'].isin(sel_customers)) &
        (df_claims['plant'].isin(sel_plants)) &
        (df_claims['status'].isin(sel_status))
    )
    df_filtered_for_select = df_claims[mask].copy()

    sel_claim_id = st.selectbox(
        "Claim ID",
        options=sorted(df_filtered_for_select['claim_id'].unique().tolist()),
        index=0 if not df_filtered_for_select.empty else None,
        disabled=df_filtered_for_select.empty
    )

# -----------------------------
# Apply Filters to Dataset
# -----------------------------
mask_main = (
    (df_claims['start_date'].dt.date.between(date_range[0], date_range[1])) &
    (df_claims['customer'].isin(sel_customers)) &
    (df_claims['plant'].isin(sel_plants)) &
    (df_claims['status'].isin(sel_status))
)

fdf = df_claims[mask_main].copy()

# -----------------------------
# Title Row
# -----------------------------
st.markdown(
    f"""
    <div class='title'>📊 Claims Executive Dashboard</div>
    <div class='subtitle'>As of {datetime.now().strftime('%b %d, %Y %H:%M')} · Filtered on Customer, Plant, Status, Start Date</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# Guard: no data after filters
if fdf.empty:
    st.warning("No claims match the filters. Adjust filters in the sidebar.")
    st.stop()

# -----------------------------
# KPI Cards
# -----------------------------
active_mask = fdf['status'].isin(['In Progress', 'Stuck'])
completed_mask = fdf['status'] == 'Completed'
cancelled_mask = fdf['status'] == 'Cancelled'

active_count = int(active_mask.sum())
completed_count = int(completed_mask.sum())
cancelled_count = int(cancelled_mask.sum())

active_value = float(fdf.loc[active_mask, 'estimated_amount'].fillna(0).sum())
paid_total = float(fdf.loc[completed_mask, 'final_paid_amount'].fillna(0).sum())
est_completed_total = float(fdf.loc[completed_mask, 'estimated_amount'].fillna(0).sum())
recovery_rate = (paid_total / est_completed_total * 100) if est_completed_total > 0 else 0.0

median_cycle = float(fdf.loc[completed_mask, 'cycle_time_days'].dropna().median()) if completed_count else 0

# Stuck >7 days detection (based on last milestone vs now)
last_ms = df_milestones.sort_values('date').groupby('claim_id')['date'].max().rename('last_ms')
fdf2 = fdf.join(last_ms, on='claim_id')
now_ts = pd.Timestamp(datetime.now())
fdf2['days_since_last_ms'] = (now_ts - fdf2['last_ms']).dt.days
stuck_over_7 = int(((fdf2['status'] == 'Stuck') & (fdf2['days_since_last_ms'] > 7)).sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi_card("Active Claims", f"{active_count} ⚠")
with c2: kpi_card("Completed", f"{completed_count} ✔")
with c3: kpi_card("Cancelled", f"{cancelled_count} ✖")
with c4: kpi_card("Value of Active", fmt_money(active_value))
with c5: kpi_card("Paid on Completed", fmt_money(paid_total))
with c6: kpi_card("Recovery Rate", f"{recovery_rate:.1f}%")

st.markdown("\n")

# -----------------------------
# Row: Claim Timeline + Claim Facts (Left)  |  Financials (Right)
# -----------------------------
left, right = st.columns([1.3, 1.7])

# ---- Left: Selected Claim Timeline & Facts
with left:
    st.subheader("Lifecycle of Selected Claim")
    if sel_claim_id:
        sel = df_claims[df_claims['claim_id'] == sel_claim_id].iloc[0]
        ms = df_milestones[df_milestones['claim_id'] == sel_claim_id].sort_values('date')

        # Claim facts box
        facts_cols = st.columns(2)
        with facts_cols[0]:
            st.write(f"**Claim ID:** {sel['claim_id']}")
            st.write(f"**Status:** {sel['status']}")
            st.write(f"**Customer:** {sel['customer']}")
            st.write(f"**Plant:** {sel['plant']}")
        with facts_cols[1]:
            st.write(f"**Estimated:** {fmt_money(sel['estimated_amount'])}")
            st.write(f"**Paid:** {fmt_money(sel['final_paid_amount'])}")
            st.write(f"**Start:** {pd.to_datetime(sel['start_date']).date()}")
            st.write(f"**Completion:** {str(pd.to_datetime(sel['completion_date']).date()) if pd.notna(sel['completion_date']) else '—'}")

        # Timeline chart
        if not ms.empty and ms['date'].notna().any():
            ms = ms.dropna(subset=['date'])
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=ms['date'], y=[1]*len(ms), mode='markers+lines+text',
                text=ms['milestone'], textposition='bottom center',
                marker=dict(size=14, color='#3949ab'),
                line=dict(color='#94a3b8', width=2)
            ))
            fig_tl.update_layout(
                title=f"Timeline — {sel['claim_id']} ({sel['status']})",
                xaxis_title="Date", yaxis_visible=False, showlegend=False,
                margin=dict(l=20, r=20, t=60, b=10)
            )
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("No milestone data for this claim.")

        # Case study callouts
        if sel['claim_id'] == 'P6W23BEG36-2025':
            last_activity = df_milestones[df_milestones['claim_id']==sel['claim_id']]['date'].max()
            days_stalled = (now_ts - last_activity).days if pd.notna(last_activity) else '—'
            st.info(f"🟠 **Stuck** · {sel['claim_id']} — {fmt_money(sel['estimated_amount'])}. Last activity: {last_activity.date() if pd.notna(last_activity) else '—'} · ~{days_stalled} days.")
        if sel['claim_id'] == 'P4W13BEG25-2025':
            st.success(f"🟢 **Completed** · {sel['claim_id']} — {fmt_money(sel['final_paid_amount'])} paid. Closed in {int(sel['cycle_time_days']) if pd.notna(sel['cycle_time_days']) else '—'} days.")

# ---- Right: Financial Waterfall + Root Causes
with right:
    st.subheader("Financial Impact Overview")
    est_total = float(fdf['estimated_amount'].fillna(0).sum())
    cancelled_amt = float(fdf.loc[cancelled_mask, 'estimated_amount'].fillna(0).sum())
    paid_amt = float(fdf.loc[completed_mask, 'final_paid_amount'].fillna(0).sum())
    # Net exposure = est_total - cancelled - paid
    measures = ["relative", "relative", "relative", "total"]
    x_labels = ["Estimated Claims", "Cancelled", "Paid", "Net Exposure"]
    y_vals = [est_total, -cancelled_amt, -paid_amt, 0]  # total computed by waterfall

    wf = go.Figure(go.Waterfall(
        name="FinancialFlow", orientation="v",
        measure=measures, x=x_labels, y=y_vals,
        text=[fmt_money(est_total), f"-{fmt_money(cancelled_amt)}", f"-{fmt_money(paid_amt)}", ""],
        connector={"line": {"color": "#475569"}}
    ))
    wf.update_layout(
        margin=dict(l=20,r=20,t=40,b=10),
        yaxis_title="USD",
        title="Claims Financial Flow",
    )
    st.plotly_chart(wf, use_container_width=True)

    # Root Cause Distribution — Estimated vs Paid by Reason
    st.subheader("Root Cause Distribution")
    rdf = fdf.explode('reason').copy()
    # Estimated impact by reason (all statuses)
    est_by_reason = rdf.groupby('reason', dropna=True)['estimated_amount'].sum().rename('Estimated').reset_index()
    # Paid impact by reason (completed only)
    paid_by_reason = rdf[rdf['status']=='Completed'].groupby('reason')['final_paid_amount'].sum().rename('Paid').reset_index()
    rc = est_by_reason.merge(paid_by_reason, on='reason', how='left').fillna(0)
    rc = rc.sort_values(by='Estimated', ascending=False)

    fig_rc = go.Figure()
    fig_rc.add_bar(x=rc['reason'], y=rc['Estimated'], name='Estimated')
    fig_rc.add_bar(x=rc['reason'], y=rc['Paid'], name='Paid')
    fig_rc.update_layout(
        barmode='group',
        title='Impact by Reason (Estimated vs Paid)',
        yaxis_title='USD', xaxis_title='Reason',
        margin=dict(l=20,r=20,t=50,b=10)
    )
    st.plotly_chart(fig_rc, use_container_width=True)

# -----------------------------
# Row: Top Entities
# -----------------------------
st.subheader("Top Entities (by Total Estimated)")
colA, colB = st.columns(2)

with colA:
    top_cust = fdf.groupby('customer')['estimated_amount'].sum().sort_values(ascending=False).head(8)
    fig_tc = px.bar(
        top_cust.reset_index(), x='customer', y='estimated_amount',
        title='Top Customers by Total Estimated', labels={'estimated_amount':'USD','customer':'Customer'},
        color='customer', color_discrete_sequence=px.colors.sequential.Blues
    )
    fig_tc.update_layout(showlegend=False, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig_tc, use_container_width=True)

with colB:
    top_plant = fdf.groupby('plant')['estimated_amount'].sum().sort_values(ascending=False).head(8)
    fig_tp = px.bar(
        top_plant.reset_index(), x='plant', y='estimated_amount',
        title='Top Plants by Total Estimated', labels={'estimated_amount':'USD','plant':'Plant'},
        color='plant', color_discrete_sequence=px.colors.sequential.Blues
    )
    fig_tp.update_layout(showlegend=False, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig_tp, use_container_width=True)

# -----------------------------
# Correlation Heatmaps (Plant × Client × Reason / Financial Hit)
# -----------------------------

st.subheader("Correlation Heatmaps — Pattern Finder")

# Helper to build pivots with optional normalization

def build_heatmap_pivot(df: pd.DataFrame, index: str, columns: str, metric: str, normalize: str = "None"):
    tmp = df.copy()
    value_col = None
    aggfunc = 'sum'

    if metric == "Count of claims":
        # Size/count pivot
        pv = pd.pivot_table(tmp, index=index, columns=columns, values='estimated_amount', aggfunc='size', fill_value=0)
    elif metric == "Estimated amount (USD)":
        value_col = 'estimated_amount'
        pv = pd.pivot_table(tmp, index=index, columns=columns, values=value_col, aggfunc='sum', fill_value=0)
    else:  # Paid amount
        value_col = 'final_paid_amount'
        # Paid only makes sense for completed claims; treat NaN as 0
        tmp_paid = tmp.copy()
        tmp_paid.loc[tmp_paid['status'] != 'Completed', 'final_paid_amount'] = 0.0
        pv = pd.pivot_table(tmp_paid, index=index, columns=columns, values=value_col, aggfunc='sum', fill_value=0)

    # Normalize if requested
    if normalize == "Row":
        pv = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == "Column":
        pv = pv.div(pv.sum(axis=0).replace(0, np.nan), axis=1)

    # Replace infinities/NaNs from normalization
    pv = pv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pv

# Explode reasons once for all heatmaps
rdf = fdf.explode('reason').dropna(subset=['reason']).copy()

# UI controls for Heatmap A (Reason × Plant)
hA_left, hA_right = st.columns([1, 1])
with hA_left:
    st.markdown("**Reason × Plant** — correlation heatmap")
    metric_A = st.selectbox(
        "Metric",
        ["Count of claims", "Estimated amount (USD)", "Paid amount (USD)"],
        index=0,
        key="hm_A_metric"
    )
    normalize_A = st.selectbox("Normalize", ["None", "Row", "Column"], index=0, key="hm_A_norm")
    # Optional: limit to top-N reasons by total to keep it readable
    topN_A = st.slider("Top reasons (by volume)", min_value=5, max_value=max(5, len(rdf['reason'].unique())), value=min(12, len(rdf['reason'].unique())), step=1, key="hm_A_top")

    # Determine top reasons
    vol_reason = rdf.groupby('reason').size().sort_values(ascending=False)
    keep_reasons = vol_reason.head(topN_A).index.tolist()
    rdf_A = rdf[rdf['reason'].isin(keep_reasons)].copy()

    pv_A = build_heatmap_pivot(rdf_A, index='reason', columns='plant', metric=metric_A, normalize=normalize_A)

    fig_A = go.Figure(data=go.Heatmap(
        z=pv_A.values,
        x=[str(c) for c in pv_A.columns],
        y=pv_A.index.astype(str),
        colorscale='Blues',
        colorbar_title=("Count" if metric_A == "Count of claims" else ("USD" if normalize_A == "None" else "% of row/col")),
        hoverongaps=False
    ))
    fig_A.update_layout(
        margin=dict(l=20, r=20, t=30, b=10),
        xaxis_title="Plant",
        yaxis_title="Reason",
        title=f"Reason × Plant — {metric_A} (normalize: {normalize_A})"
    )
    st.plotly_chart(fig_A, use_container_width=True)

with hA_right:
    st.markdown("**Reason × Customer** — correlation heatmap")
    metric_B = st.selectbox(
        "Metric",
        ["Count of claims", "Estimated amount (USD)", "Paid amount (USD)"],
        index=0,
        key="hm_B_metric"
    )
    normalize_B = st.selectbox("Normalize", ["None", "Row", "Column"], index=0, key="hm_B_norm")
    topN_B = st.slider("Top reasons (by volume)", min_value=5, max_value=max(5, len(rdf['reason'].unique())), value=min(12, len(rdf['reason'].unique())), step=1, key="hm_B_top")

    vol_reason_B = rdf.groupby('reason').size().sort_values(ascending=False)
    keep_reasons_B = vol_reason_B.head(topN_B).index.tolist()
    rdf_B = rdf[rdf['reason'].isin(keep_reasons_B)].copy()

    pv_B = build_heatmap_pivot(rdf_B, index='reason', columns='customer', metric=metric_B, normalize=normalize_B)

    fig_B = go.Figure(data=go.Heatmap(
        z=pv_B.values,
        x=[str(c) for c in pv_B.columns],
        y=pv_B.index.astype(str),
        colorscale='Blues',
        colorbar_title=("Count" if metric_B == "Count of claims" else ("USD" if normalize_B == "None" else "% of row/col")),
        hoverongaps=False
    ))
    fig_B.update_layout(
        margin=dict(l=20, r=20, t=30, b=10),
        xaxis_title="Customer",
        yaxis_title="Reason",
        title=f"Reason × Customer — {metric_B} (normalize: {normalize_B})"
    )
    st.plotly_chart(fig_B, use_container_width=True)

# UI controls for Heatmap C (Plant × Customer — Financial Hit)

st.markdown("---")
st.markdown("**Plant × Customer — financial hit heatmap**")
metric_C = st.selectbox(
    "Metric",
    ["Estimated amount (USD)", "Paid amount (USD)", "Count of claims"],
    index=0,
    key="hm_C_metric"
)
normalize_C = st.selectbox("Normalize", ["None", "Row", "Column"], index=0, key="hm_C_norm")

if metric_C == "Count of claims":
    pv_C = pd.pivot_table(fdf, index='plant', columns='customer', values='estimated_amount', aggfunc='size', fill_value=0)
elif metric_C == "Estimated amount (USD)":
    pv_C = pd.pivot_table(fdf, index='plant', columns='customer', values='estimated_amount', aggfunc='sum', fill_value=0)
else:
    tmpC = fdf.copy()
    tmpC.loc[tmpC['status'] != 'Completed', 'final_paid_amount'] = 0.0
    pv_C = pd.pivot_table(tmpC, index='plant', columns='customer', values='final_paid_amount', aggfunc='sum', fill_value=0)

if normalize_C == "Row":
    pv_C = pv_C.div(pv_C.sum(axis=1).replace(0, np.nan), axis=0)
elif normalize_C == "Column":
    pv_C = pv_C.div(pv_C.sum(axis=0).replace(0, np.nan), axis=1)

pv_C = pv_C.replace([np.inf, -np.inf], np.nan).fillna(0)

fig_C = go.Figure(data=go.Heatmap(
    z=pv_C.values,
    x=[str(c) for c in pv_C.columns],
    y=pv_C.index.astype(str),
    colorscale='Blues',
    colorbar_title=("Count" if metric_C == "Count of claims" else ("USD" if normalize_C == "None" else "% of row/col")),
    hoverongaps=False
))
fig_C.update_layout(
    margin=dict(l=20, r=20, t=30, b=10),
    xaxis_title="Customer",
    yaxis_title="Plant",
    title=f"Plant × Customer — {metric_C} (normalize: {normalize_C})"
)
st.plotly_chart(fig_C, use_container_width=True)

# -----------------------------
# Claims Table (filtered)
# -----------------------------
st.subheader("Claims Table (Filtered)")
show_cols = ['claim_id','customer','plant','status','estimated_amount','final_paid_amount','start_date','completion_date','days_open','cycle_time_days']

table_df = fdf[show_cols].copy()
table_df = table_df.sort_values(by=['status','start_date'], ascending=[True, False])

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)

# Allow CSV download of the filtered claims
csv = table_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download filtered claims as CSV",
    data=csv,
    file_name="claims_filtered.csv",
    mime="text/csv",
)
