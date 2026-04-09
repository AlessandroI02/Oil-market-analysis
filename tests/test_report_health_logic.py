import pandas as pd

from src.report_writer import build_insights


def test_insights_suppress_recommendations_when_invalid_run():
    core = pd.DataFrame({"ticker": ["A"], "final_score": [70], "publishable_flag": [False]})
    ext = pd.DataFrame({"ticker": ["A"], "final_score": [70]})

    insights = build_insights(
        core_ranking_df=core,
        extended_ranking_df=ext,
        valuation_df=pd.DataFrame(),
        scenario_df=pd.DataFrame(),
        factor_df=pd.DataFrame(),
        analogue_df=pd.DataFrame(),
        catalyst_df=pd.DataFrame(),
        recommendation_df=pd.DataFrame({"category": ["Best"], "ticker": ["A"]}),
        run_summary={"run_status": "INVALID"},
        section_health_df=pd.DataFrame(
            {"section_name": ["recommendations"], "section_status": ["INVALID"], "reason": ["No data"]}
        ),
    )

    assert insights.run_status == "INVALID"
    assert insights.recommendation_allowed is False
    assert "INVALID" in insights.executive_summary
