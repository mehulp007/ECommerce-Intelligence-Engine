"""Response and request models; they also define the OpenAPI contract."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

RiskBand = Literal["low", "medium", "high", "lapsed"]
Reason = Literal[
    "buy_again",
    "bought_together",
    "similar_style",
    "similar_shoppers",
    "popular_in_age_group",
    "trending",
]


class Article(BaseModel):
    article_id: str
    prod_name: str | None
    product_type_name: str | None
    product_group_name: str | None
    colour_group_name: str | None
    department_name: str | None
    index_group_name: str | None
    section_name: str | None
    garment_group_name: str | None
    detail_desc: str | None
    median_price: float | None = None
    sales_4w: int | None = None
    popularity_rank_4w: int | None = None
    has_image: bool


class ArticleDetail(Article):
    sales_total: int | None
    sales_1w: int | None
    buyers_total: int | None
    first_sale: date | None
    last_sale: date | None
    online_share: float | None
    weekly_sales_12w: list[int]


class ArticlePage(BaseModel):
    total: int
    items: list[Article]


class Facets(BaseModel):
    index_groups: list[str]
    product_groups: list[str]
    colours: list[str]


class SearchHit(BaseModel):
    score: float
    article: Article


class Recommendation(BaseModel):
    rank: int
    article_id: str
    score: float
    reason: Reason
    article: Article | None


class Driver(BaseModel):
    feature: str
    value: float | None
    log_odds: float = Field(description="Contribution to the raw model's log-odds")


class CustomerProfile(BaseModel):
    age: int | None = None
    club_member_status: str | None = None
    fashion_news_frequency: str | None = None
    active: bool | None = None
    fashion_news_subscriber: bool | None = None


class CustomerSummary(BaseModel):
    purchases: int
    spend_units: float | None
    visits: int
    first_purchase: date | None
    last_purchase: date | None
    online_share: float | None


class Customer(BaseModel):
    customer_id: str
    churn_probability: float | None = Field(
        description="Calibrated probability of no purchase in the next 60 days"
    )
    risk_band: RiskBand
    drivers: list[Driver]
    segment_id: int
    segment_name: str
    recency_days: float | None
    visits_365d: float | None
    spend_365d: float | None
    p_alive: float | None
    expected_visits_180d: float | None
    expected_visit_spend: float | None
    clv_180d: float | None
    profile: CustomerProfile
    summary: CustomerSummary


class Purchase(BaseModel):
    purchase_date: date
    article_id: str
    price: float
    channel: Literal["online", "store"]
    article: Article | None


class CustomerSample(BaseModel):
    customer_id: str
    churn_probability: float
    risk_band: RiskBand
    segment_id: int
    segment_name: str
    clv_180d: float | None
    visits_365d: float | None
    spend_365d: float | None
    recency_days: float | None


class Segment(BaseModel):
    segment_id: int
    segment_name: str
    customers: int
    mean_churn_probability: float | None
    median_clv_180d: float | None
    total_clv_180d: float | None
    median_visits_365d: float | None
    median_spend_365d: float | None


class CampaignRequest(BaseModel):
    budget: float = Field(500.0, ge=0, description="Total budget in price units")
    cost_per_contact: float = Field(0.005, gt=0)
    uplift: float = Field(0.10, gt=0, le=1, description="Assumed retention uplift")
    segment_names: list[str] | None = None
    risk_bands: list[Literal["low", "medium", "high"]] | None = None
    top: int = Field(25, ge=0, le=500)


class CampaignCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    clv_180d: float
    segment_name: str
    risk_band: RiskBand
    expected_value_saved: float


class CurvePoint(BaseModel):
    contacted: int
    spend: float
    expected_value_saved: float
    expected_net: float


class CampaignPlan(BaseModel):
    eligible_customers: int
    contacted: int
    spend: float
    expected_value_saved: float
    expected_net: float
    roi: float
    profitable_customers: int
    contacted_by_segment: dict[str, int]
    top_customers: list[CampaignCustomer]
    curve: list[CurvePoint]


class Health(BaseModel):
    status: Literal["ready"]
    version: str
    data_end: str
    built_at: str
    counts: dict[str, int]
    text_search: bool
