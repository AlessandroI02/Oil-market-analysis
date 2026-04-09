# Repo 1 Handoff Contract

## Scope Boundary
Repo 1 is the upstream market-intelligence engine.

Repo 1 owns:
- market and factor data
- route/chokepoint intelligence
- event/news episodes
- regime classification
- market-side statistical math
- market-to-valuation overlays
- confidence framework and packet readiness flags
- company case packets and schema-locked exports

Repo 1 does **not** own:
- full DCF/WACC/terminal-value engines
- final target-price outputs
- final investment-conviction recommendations

## Date And Identifier Rules
- Dates use ISO `YYYY-MM-DD` (`as_of_date`, `event_date`, `start_date`, `end_date`).
- `ticker` uses repo canonical ticker symbols (for example `XOM`, `REP.MC`, `0857.HK`).
- `company_name` uses universe canonical display name.

## Confidence Contract
Confidence is multidimensional and intentionally conservative.

Core fields:
- `input_data_confidence`
- `source_confidence`
- `route_model_confidence`
- `event_model_confidence`
- `regime_model_confidence`
- `packet_confidence`
- `downstream_readiness_flag`

Compatibility fields remain in packets:
- `data_confidence`
- `route_confidence`
- `event_confidence`
- `regime_confidence`
- `company_packet_confidence`

Alignment rule:
- if ranking status is `unrated` or `publishable_flag=False` with `final_rating_confidence=Low`, packet confidence is capped and `downstream_readiness_flag=False`.

Audit file:
- `outputs/debug/confidence_audit.csv`

## Regime Labels
- `normal`
- `stressed oil regime`
- `geopolitical supply-shock regime`
- `sanctions-driven regime`
- `route-disruption regime`
- `macro risk-off regime`
- `mixed / transition regime`

## Market Constraint Contract
`market_constraints.csv` is a **heuristic overlay** for downstream valuation workflows.
It is not a valuation engine.

Key fields:
- `suggested_discount_rate_uplift_bps`
- `suggested_discount_rate_uplift_range_bps`
- `suggested_beta_adjustment`
- `suggested_beta_adjustment_range`
- `suggested_risk_premium_bucket`
- `scenario_probability_shift`
- `geopolitical_stress_multiplier`
- `confidence_penalty_recommendation`
- `constraint_confidence`
- `methodology_label`
- `market_regime_impact_note`

Methodology audit:
- `outputs/debug/market_constraints_methodology.csv`

## Core Output Paths
- `outputs/debug/regime_state.csv`
- `outputs/debug/regime_history.csv`
- `outputs/debug/event_episodes.csv`
- `outputs/debug/event_episode_articles.csv`
- `outputs/debug/market_constraints.csv`
- `outputs/debug/market_constraints_methodology.csv`
- `outputs/debug/confidence_framework.csv`
- `outputs/debug/confidence_audit.csv`
- `outputs/debug/peer_baskets.csv`
- `outputs/debug/peer_basket_membership.csv`
- `outputs/debug/company_case_packet_index.csv`
- `outputs/packets/company_case_packets/{ticker}.json`

Note:
- `company_case_packet_index.csv` stores repo-relative packet paths where practical.

## Market Math Exports
- `outputs/debug/returns_daily_matrix.csv`
- `outputs/debug/returns_weekly_matrix.csv`
- `outputs/debug/cumulative_returns_matrix.csv`
- `outputs/debug/benchmark_returns_matrix.csv`
- `outputs/debug/correlation_matrix.csv`
- `outputs/debug/rolling_correlations.csv`
- `outputs/debug/regression_coefficients.csv`
- `outputs/debug/regression_diagnostics.csv`
- `outputs/debug/rolling_betas.csv`
- `outputs/debug/event_study_summary.csv`
- `outputs/debug/event_study_detail.csv`
- `outputs/debug/cluster_assignments.csv`
- `outputs/debug/metric_zscores.csv`
- `outputs/debug/metric_percentiles.csv`

## News/Event Source Discipline Outputs
- `outputs/debug/oil_news_articles.csv`
- `outputs/debug/oil_news_event_days.csv`
- `outputs/debug/oil_news_catalysts.csv`
- `outputs/debug/oil_news_episode_summary.csv`

## Catalyst Outputs
- `outputs/debug/catalyst_calendar_primary.csv`
- `outputs/debug/catalyst_calendar_archive.csv`

Primary list intent:
- high/medium relevance future events with cleaner confidence and clutter reduction.

## Downstream Consumption Guidance
Repo 2 / Repo 3 should:
- treat Repo 1 constraints as scenario overlays, not final valuation assumptions
- use `downstream_readiness_flag` and `constraint_confidence` as gating metadata
- preserve packet confidence caps when ranking health is weak
- use episode summaries and source tiers as evidence context, not causal proof

## Schema Files
- `schemas/company_case_packet.schema.json`
- `schemas/market_constraints.schema.json`
- `schemas/regime_state.schema.json`
- `schemas/event_episode.schema.json`

## Units
- Returns/sensitivities: percentages unless otherwise labeled.
- Betas: unitless.
- Discount-rate uplift: basis points.
