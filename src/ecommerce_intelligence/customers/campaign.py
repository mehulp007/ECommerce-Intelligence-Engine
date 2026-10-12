"""Retention campaign simulator: who to contact and the expected return.

For a customer with inactivity probability p and 180-day value V, a contact
costing c that retains a would-be-inactive customer with probability u is worth
p * u * V - c in expectation. Values are in H&M's normalized price units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignPlan:
    contacted: int
    spend: float
    expected_value_saved: float
    expected_net: float
    roi: float
    customers: pd.DataFrame


def expected_value(scores: pd.DataFrame, uplift: float) -> np.ndarray:
    return (
        scores["churn_probability"].to_numpy(float)
        * uplift
        * scores["clv_180d"].to_numpy(float)
    )


def plan(
    scores: pd.DataFrame,
    *,
    budget: float,
    cost_per_contact: float,
    uplift: float,
    profitable_only: bool = True,
) -> CampaignPlan:
    """Contact the customers with the highest expected saved value."""
    if budget < 0 or cost_per_contact <= 0 or not 0 < uplift <= 1:
        raise ValueError("Need budget >= 0, cost > 0, and 0 < uplift <= 1")
    eligible = scores.dropna(subset=["churn_probability", "clv_180d"])
    value = expected_value(eligible, uplift)
    order = np.argsort(-value, kind="stable")
    # A tiny tolerance stops float division (500 / 0.005 = 99999.99...) losing one.
    limit = min(int(np.floor(budget / cost_per_contact + 1e-9)), len(eligible))
    if profitable_only:
        limit = min(limit, int((value > cost_per_contact).sum()))
    chosen = order[:limit]
    saved = float(value[chosen].sum())
    spend = limit * cost_per_contact
    customers = eligible.iloc[chosen].assign(expected_value_saved=value[chosen])
    return CampaignPlan(
        contacted=limit,
        spend=spend,
        expected_value_saved=saved,
        expected_net=saved - spend,
        roi=(saved - spend) / spend if spend else 0.0,
        customers=customers,
    )


def roi_curve(
    scores: pd.DataFrame,
    *,
    cost_per_contact: float,
    uplift: float,
    points: int = 50,
) -> pd.DataFrame:
    """Cumulative spend, saved value, and net return as more customers are contacted."""
    value = np.sort(
        expected_value(scores.dropna(subset=["churn_probability", "clv_180d"]), uplift)
    )[::-1]
    sizes = np.unique(np.linspace(0, len(value), points + 1).astype(int))[1:]
    cumulative = np.cumsum(value)
    saved = cumulative[sizes - 1]
    spend = sizes * cost_per_contact
    return pd.DataFrame(
        {
            "contacted": sizes,
            "spend": spend,
            "expected_value_saved": saved,
            "expected_net": saved - spend,
        }
    )
