"""Time windows and sizes shared by every recommender stage."""

from __future__ import annotations

from datetime import date, timedelta

K = 12
WEEK = timedelta(days=7)

# The final observed week, 2020-09-16 to 2020-09-22, is the held-out evaluation
# week, matching the last week of the competition's public data.
VALIDATION_START = date(2020, 9, 16)
# Production recommendations are made as of the day after the data ends.
INFERENCE_AS_OF = VALIDATION_START + WEEK
TRAINING_WEEKS = 3

REPURCHASE_DAYS = 84
REPURCHASE_CANDIDATES = 40
POPULAR_CANDIDATES = 60
AGE_POPULAR_CANDIDATES = 30
COOC_DAYS = 56
COOC_BASKET_LIMIT = 30
COOC_PARTNERS = 10
COOC_SEEDS = 20
COOC_CANDIDATES = 40
ALS_DAYS = 84
ALS_CANDIDATES = 40
STYLE_SEEDS = 20
STYLE_ACTIVE_DAYS = 28
STYLE_CANDIDATES = 40

SOURCES = ("repurchase", "popular", "age_popular", "cooc", "als", "style")
# Reason shown to a shopper, in priority order when several sources agree.
REASONS = {
    "repurchase": "buy_again",
    "cooc": "bought_together",
    "style": "similar_style",
    "als": "similar_shoppers",
    "age_popular": "popular_in_age_group",
    "popular": "trending",
}


def training_as_of_dates(last_label_week_start: date, weeks: int) -> list[date]:
    """As-of dates whose following week supplies training labels."""
    return [last_label_week_start - WEEK * offset for offset in range(weeks)]
