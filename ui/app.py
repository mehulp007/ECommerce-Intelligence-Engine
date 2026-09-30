"""Historical customer intelligence dashboard backed by the FastAPI service."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ui.client import ApiClient, ApiError

st.set_page_config(
    page_title="E-Commerce Intelligence Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PERSONA_DESCRIPTIONS = {
    "At-risk": "Longer time since the last purchase; an opportunity to re-engage.",
    "High-value": "Historically higher spend across the observed purchase window.",
    "Repeat buyers": "Frequent purchasing activity in the observed history.",
    "Occasional": "Lighter purchase history with room to build a habit.",
}
FEATURE_LABELS = {
    "recency_days": "Days since purchase",
    "frequency_180d": "Orders · 180 days",
    "monetary_180d": "Spend · 180 days",
    "orders_30d": "Orders · 30 days",
    "spend_30d": "Spend · 30 days",
    "distinct_products_180d": "Distinct products",
    "return_invoices_180d": "Return invoices",
}


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except FileNotFoundError:
        return None
    return str(value) if value else None


def _client() -> ApiClient | None:
    url = _secret("API_URL") or os.getenv("EIE_API_URL")
    key = _secret("API_KEY") or os.getenv("EIE_API_KEY")
    if not url or not key:
        return None
    return ApiClient(base_url=url, api_key=key)


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def _section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-heading"><h2>{_safe(title)}</h2><p>{_safe(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def _metric(label: str, value: str, foot: str, *, tone: str = "") -> None:
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-label">{_safe(label)}</div>'
        f'<div class="metric-value {tone}">{_safe(value)}</div>'
        f'<div class="metric-foot">{_safe(foot)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _contribution_chart(drivers: list[dict]) -> go.Figure:
    ordered = list(reversed(drivers))
    values = [item["log_odds_contribution"] for item in ordered]
    labels = [FEATURE_LABELS.get(item["feature"], item["feature"]) for item in ordered]
    colors = ["#be705a" if value >= 0 else "#268a79" for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            customdata=[item["value"] for item in ordered],
            hovertemplate="%{y}<br>Value: %{customdata:,.2f}<br>Raw log-odds contribution: %{x:+.3f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#c9d8d2")
    figure.update_layout(
        height=340,
        margin={"l": 22, "r": 22, "t": 22, "b": 32},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        xaxis_title="Effect on log-odds",
        font={"family": "DM Sans, sans-serif", "color": "#18312f", "size": 12},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#ecf1ee", zeroline=False)
    figure.update_yaxes(showgrid=False)
    return figure


def _product_card(product: dict, rank: int) -> None:
    score = product.get("score")
    source = (
        f"BM25 relevance {score:.2f}" if score is not None else "Popular before cutoff"
    )
    st.markdown(
        '<div class="product-card">'
        '<div class="product-head">'
        f'<div><div class="product-title">{rank:02d} · {_safe(product["description"])}</div>'
        f'<div class="product-code">Stock code {_safe(product["stock_code"])}</div></div>'
        f'<div class="product-price">£{product["median_unit_price"]:,.2f}</div>'
        "</div>"
        f'<div class="product-foot">{_safe(source)} · Historical median price</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="topbar"><div class="brand"><span class="brand-mark"></span>'
        'E-Commerce Intelligence Engine</div><div class="topbar-note">'
        "Historical retail intelligence · Portfolio demo</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="eyebrow">Customer intelligence workspace</div>',
        unsafe_allow_html=True,
    )
    st.title("Know the customer behind the transaction.")
    st.markdown(
        '<div class="hero-subtitle">Explore a historical customer profile, a calibrated '
        "30-day inactivity proxy, the model's directional drivers, and products they "
        "have not already bought.</div>",
        unsafe_allow_html=True,
    )

    client = _client()
    if client is None:
        st.error(
            "Dashboard connection is not configured. Set API_URL and API_KEY as "
            "Streamlit secrets, or EIE_API_URL and EIE_API_KEY in the local server environment."
        )
        st.stop()

    try:
        health = client.health()
        demo = client.demo_customers()
    except ApiError as error:
        st.error(str(error))
        if st.button("Retry connection", type="primary"):
            st.rerun()
        st.stop()

    st.markdown(
        '<div class="hero-meta"><span class="dot"></span>'
        f'Historical snapshot {_safe(demo["as_of"])} · Bundle {_safe(health["bundle_version"])}'
        "</div>",
        unsafe_allow_html=True,
    )

    _section(
        "Customer lookup",
        "Choose a verified sample or enter a customer ID from this historical cohort.",
    )
    with st.form("customer_lookup", clear_on_submit=False):
        sample_col, input_col = st.columns([1, 1])
        with sample_col:
            sample_id = st.selectbox(
                "Verified sample ID",
                ["Choose a sample", *demo["customer_ids"]],
            )
        with input_col:
            entered_id = st.text_input(
                "Or enter a customer ID", placeholder="Numeric customer ID"
            )
        st.caption(
            "A selected sample takes precedence. Choose the first option to use a typed ID."
        )
        submitted = st.form_submit_button("View customer intelligence", type="primary")

    if submitted:
        customer_id = (
            sample_id if sample_id != "Choose a sample" else entered_id.strip()
        )
        if not re.fullmatch(r"[0-9]{1,20}", customer_id):
            st.warning("Select a sample or enter a numeric customer ID.")
            st.session_state.pop("customer_result", None)
        else:
            st.session_state.pop("customer_result", None)
            try:
                with st.spinner("Loading historical customer intelligence…"):
                    prediction = client.prediction(customer_id)
                    recommendations = client.recommendations(customer_id)
                st.session_state["customer_result"] = (prediction, recommendations)
            except ApiError as error:
                st.error(str(error))

    result = st.session_state.get("customer_result")
    if result is None:
        st.markdown(
            '<div class="empty-card">Select a sample customer above to see the '
            "complete profile and product recommendations.</div>",
            unsafe_allow_html=True,
        )
        return

    prediction, recommendations = result
    persona_name = prediction["persona"]["name"]
    probability = prediction["inactivity_probability_30d"]
    risk = prediction["risk_band"]
    _section(
        f'Customer {prediction["customer_id"]}',
        "One fixed historical view · Features from purchases before the snapshot date.",
    )
    metric_columns = st.columns(4)
    with metric_columns[0]:
        _metric(
            "30-day inactivity proxy",
            f"{probability:.1%}",
            f"{risk.capitalize()} risk · calibrated probability",
            tone=f"risk-{risk}",
        )
    with metric_columns[1]:
        _metric("Customer persona", persona_name, "RFM-based segment")
    with metric_columns[2]:
        _metric(
            "Suggested products",
            str(len(recommendations["products"])),
            "Previously bought items excluded",
        )
    with metric_columns[3]:
        _metric(
            "As-of date", prediction["as_of"], "Historical snapshot · not live data"
        )

    _section(
        "Why the model leaned this way",
        "Directional contributions from the raw XGBoost model.",
    )
    st.plotly_chart(
        _contribution_chart(prediction["drivers"]),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )
    st.markdown(
        '<div class="fine-print">Coral bars push the raw model toward inactivity; '
        "teal bars push toward activity. These log-odds contributions do not add up "
        "to the calibrated probability and are not causal explanations.</div>",
        unsafe_allow_html=True,
    )

    _section(
        "Next product suggestions", "Ranked using purchases before November 1, 2011."
    )
    products = recommendations["products"]
    for index in range(0, len(products), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            product_index = index + offset
            if product_index < len(products):
                with column:
                    _product_card(products[product_index], product_index + 1)

    _section(
        "Persona guide",
        "Four behavior groups fitted on the August 2011 customer cohort.",
    )
    for column, (name, description) in zip(
        st.columns(4), PERSONA_DESCRIPTIONS.items(), strict=True
    ):
        active = " active" if name == persona_name else ""
        with column:
            st.markdown(
                f'<div class="persona-card{active}"><div class="persona-title">{_safe(name)}</div>'
                f'<div class="persona-copy">{_safe(description)}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="fine-print">Source: UCI Online Retail (2010–2011), Daqing Chen, '
        "CC BY 4.0. This is a historical portfolio demonstration. The target measures "
        "purchase inactivity, not confirmed account cancellation. Prices reflect "
        "historical transactions, not current inventory or offers.</div>",
        unsafe_allow_html=True,
    )


main()
