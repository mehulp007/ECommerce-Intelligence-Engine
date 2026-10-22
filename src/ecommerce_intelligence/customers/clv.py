"""Probabilistic customer lifetime value: BG/NBD visits and Gamma-Gamma spend.

Implements the closed-form likelihoods of Fader, Hardie and Lee, "Counting Your
Customers the Easy Way" (2005) and "RFM and CLV" (2005). Time is in weeks; a
"transaction" is a distinct purchase day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

HORIZON_DAYS = 180


def rfm_summary(con: duckdb.DuckDBPyConnection, as_of: str) -> pd.DataFrame:
    """Repeat visits x, last-repeat time t_x and age T per customer, in weeks."""
    frame = con.execute(f"""
        WITH visits AS (
            SELECT customer_idx, purchase_date, sum(price) AS spend
            FROM tx WHERE purchase_date < DATE '{as_of}'
            GROUP BY customer_idx, purchase_date
        )
        SELECT customer_idx,
               count(*) - 1 AS frequency,
               date_diff('day', min(purchase_date), max(purchase_date)) / 7.0
                   AS recency,
               date_diff('day', min(purchase_date), DATE '{as_of}') / 7.0 AS age,
               avg(spend) AS monetary
        FROM visits GROUP BY customer_idx ORDER BY customer_idx
        """).df()
    frame["customer_idx"] = frame["customer_idx"].astype(np.int32)
    return frame


def _minimise(objective, x0, args, bounds, label: str) -> np.ndarray:
    """Bounded maximum likelihood, robust to line-search stops near the optimum.

    L-BFGS-B can end with ABNORMAL when rounding noise (for example from
    parallel sums) defeats its line search; a derivative-free Powell polish
    from the same point settles it. A real failure still raises.
    """
    result = minimize(objective, x0=x0, args=args, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        result = minimize(
            objective, x0=result.x, args=args, method="Powell", bounds=bounds
        )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"{label} fit failed: {result.message}")
    return result.x


@dataclass
class BetaGeoModel:
    r: float
    alpha: float
    a: float
    b: float

    @staticmethod
    def _negative_ll(log_params, x, t_x, T) -> float:
        r, alpha, a, b = np.exp(log_params)
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = betaln(a, b + x) - betaln(a, b)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(
                x > 0,
                np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + t_x),
                -np.inf,
            )
        # The second term covers customers who may have dropped out after
        # their last repeat purchase (x > 0).
        return -np.sum(a1 + a2 + np.logaddexp(a3, a4))

    @classmethod
    def fit(cls, summary: pd.DataFrame) -> BetaGeoModel:
        x = summary["frequency"].to_numpy(float)
        t_x = summary["recency"].to_numpy(float)
        T = summary["age"].to_numpy(float)
        params = _minimise(
            cls._negative_ll,
            np.log([0.5, 5.0, 0.5, 2.0]),
            (x, t_x, T),
            [(-8, 6)] * 4,
            "BG/NBD",
        )
        return cls(*np.exp(params))

    def p_alive(self, x, t_x, T) -> np.ndarray:
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = np.where(
                x > 0,
                self.a
                / (self.b + x - 1)
                * ((self.alpha + T) / (self.alpha + t_x)) ** (self.r + x),
                0.0,
            )
        return 1.0 / (1.0 + odds)

    def expected_visits(self, t: float, x, t_x, T) -> np.ndarray:
        """Expected purchase days in the next t weeks given each history."""
        # The closed form is singular only at a = 1; it holds for a < 1 too.
        if np.isclose(self.a, 1.0):
            raise ValueError("BG/NBD conditional expectation is undefined at a = 1")
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyper = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        first = (a + b + x - 1) / (a - 1)
        second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyper
        return first * second * self.p_alive(x, t_x, T)


@dataclass
class GammaGammaModel:
    p: float
    q: float
    v: float

    @staticmethod
    def _negative_ll(log_params, n, m) -> float:
        p, q, v = np.exp(log_params)
        ll = (
            gammaln(p * n + q)
            - gammaln(p * n)
            - gammaln(q)
            + q * np.log(v)
            + (p * n - 1) * np.log(m)
            + p * n * np.log(n)
            - (p * n + q) * np.log(v + n * m)
        )
        return -np.sum(ll)

    @classmethod
    def fit(cls, summary: pd.DataFrame) -> GammaGammaModel:
        repeat = summary.loc[summary["frequency"] > 0]
        n = repeat["frequency"].to_numpy(float) + 1
        m = repeat["monetary"].to_numpy(float)
        params = _minimise(
            cls._negative_ll,
            np.log([2.0, 3.0, 0.1]),
            (n, m),
            [(-8, 8)] * 3,
            "Gamma-Gamma",
        )
        return cls(*np.exp(params))

    def expected_spend(self, frequency, monetary) -> np.ndarray:
        """Expected spend per visit; one-visit customers get the population mean."""
        n = np.asarray(frequency, float) + 1
        m = np.asarray(monetary, float)
        population = self.p * self.v / (self.q - 1)
        conditional = self.p * (self.v + n * m) / (self.p * n + self.q - 1)
        return np.where(n > 1, conditional, population)


@dataclass
class LifetimeValueModel:
    visits: BetaGeoModel
    spend: GammaGammaModel

    @classmethod
    def fit(cls, summary: pd.DataFrame) -> LifetimeValueModel:
        return cls(BetaGeoModel.fit(summary), GammaGammaModel.fit(summary))

    def predict(self, summary: pd.DataFrame, days: int = HORIZON_DAYS) -> pd.DataFrame:
        x, t_x, T = (
            summary[c].to_numpy(float) for c in ("frequency", "recency", "age")
        )
        visits = self.visits.expected_visits(days / 7.0, x, t_x, T)
        spend = self.spend.expected_spend(summary["frequency"], summary["monetary"])
        return pd.DataFrame(
            {
                "customer_idx": summary["customer_idx"].to_numpy(),
                "p_alive": self.visits.p_alive(x, t_x, T),
                "expected_visits": visits,
                "expected_visit_spend": spend,
                "clv": visits * spend,
            }
        )

    def params(self) -> dict:
        return {"bg_nbd": asdict(self.visits), "gamma_gamma": asdict(self.spend)}
