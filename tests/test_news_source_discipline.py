import pandas as pd

from src.oil_news_report import _dedupe_articles, _source_tier


def test_source_tier_uses_publication_when_domain_is_wrapper():
    assert _source_tier("news.google.com", "Reuters") == "Tier 1"
    assert _source_tier("news.google.com", "Bloomberg.com") == "Tier 1"


def test_dedupe_prefers_higher_quality_source_for_same_headline():
    df = pd.DataFrame(
        [
            {
                "headline": "OPEC signals new output policy",
                "publication": "Random Blog",
                "domain": "randomblog.example",
                "canonical_url": "https://randomblog.example/x",
                "source_quality_tier": "Tier 4",
                "article_date": "2026-04-05",
            },
            {
                "headline": "OPEC signals new output policy",
                "publication": "Reuters",
                "domain": "reuters.com",
                "canonical_url": "https://reuters.com/x",
                "source_quality_tier": "Tier 1",
                "article_date": "2026-04-06",
            },
            {
                "headline": "OPEC signals new output policy",
                "publication": "Reuters",
                "domain": "reuters.com",
                "canonical_url": "https://reuters.com/x",
                "source_quality_tier": "Tier 1",
                "article_date": "2026-04-04",
            },
        ]
    )

    out = _dedupe_articles(df)
    assert len(out) == 1
    assert out.iloc[0]["publication"] == "Reuters"
