# Hormuz Integrated Oil Research System V2.3

Institutional-grade local Python research pipeline for evaluating listed integrated oil equities as expressions of a prolonged Strait of Hormuz disruption.

Repo boundary: this project is Repo 1 (upstream market intelligence). It is not a full valuation engine and not a final investment-conviction engine.

## What V2 Adds

- Strict core vs secondary universe separation in rankings and conclusions
- Archetype classification (supermajor, national champion, upstream-heavy, near-match, etc.)
- Route/chokepoint build-up with point estimate + plausible exposure range + confidence
- Valuation and shareholder-return layer (EV/EBITDA, P/E, FCF/dividend/buyback yield, leverage)
- Scenario analysis (Brent ladder + event-path disruption scenarios)
- Factor decomposition (oil, market, sector, rates, FX proxies + residual)
- Historical analogue summary and catalyst calendar
- Confidence penalty in final ranking for proxy-heavy names
- Explicit `Data_Quality` table and proxy-burden penalty integrated into score construction
- Analyst override layer (`config/analyst_overrides.yaml`)
- Expanded Excel/Word outputs and richer debug exports
- Run-health gate with `VALID / DEGRADED / INVALID` statuses
- Structured provider fallback diagnostics (`fetch_attempts.csv`, `provider_usage.csv`)
- Section/sheet/chart/ranking health diagnostics with export suppression rules
- Separate trailing-1M oil-market event/news attribution report
- Explicit regime engine and regime history outputs
- Event episode clustering (episode summary + episode article mapping)
- Market-to-valuation constraint overlays for downstream valuation repos
- Modelling-ready market math exports (returns/correlations/regressions/rolling betas/event study/clustering/zscores)
- Multidimensional confidence framework
- Explicit peer-basket definitions and membership outputs
- Company-level case packets (JSON per ticker + packet index CSV)
- JSON-schema handoff contract for downstream repos
- Final cleanup controls: confidence alignment audit, heuristic constraints methodology table, stricter catalyst prioritization, publication-aware news tiering, and portable packet-index paths

## Quick Start (VS Code)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

- Universe review only:

```powershell
python main.py --only-review
```

- Full V2 pipeline (Excel + Word):

```powershell
python main.py
```

- Skip Word:

```powershell
python main.py --skip-word
```

- Rebuild cache:

```powershell
python main.py --rebuild-cache
```

## CLI Flags

- `--start-date YYYY-MM-DD`
- `--end-date YYYY-MM-DD`
- `--lookback-months INT`
- `--frequency STR`
- `--output-dir PATH`
- `--max-companies INT`
- `--rebuild-cache`
- `--skip-word`
- `--skip-excel`
- `--only-universe`
- `--only-review`
- `--debug`

## Core Config Files

- `config/settings.yaml`: runtime defaults + V2 weights/penalties/scenario levels
- `config/scenario_settings.yaml`: analogue periods + macro events
- `config/analyst_overrides.yaml`: manual override layer

## Main Outputs

- Excel: `outputs/excel/hormuz_integrated_oil_analysis_YYYYMMDD_HHMM.xlsx`
- Word: `outputs/word/hormuz_integrated_oil_thesis_YYYYMMDD_HHMM.docx`
- Oil news Word: `outputs/word/oil_market_news_1m_YYYYMMDD_HHMM.docx`
- Charts: `outputs/charts/*.png`
- Debug CSVs: `outputs/debug/*.csv`
- Company packets: `outputs/packets/company_case_packets/{ticker}.json`
- Schemas: `schemas/*.schema.json`
- Handoff docs: `docs/repo1_handoff_contract.md`

Key debug files include:

- `company_universe_review.csv`
- `included_companies.csv` / `rejected_companies.csv`
- `hormuz_exposure_estimates.csv`
- `route_exposure_build.csv`
- `chokepoint_exposure.csv`
- `valuation.csv`
- `scenario_analysis.csv`
- `factor_decomposition.csv`
- `historical_analogues.csv`
- `catalyst_calendar.csv`
- `core_ranking.csv` / `extended_ranking.csv`
- `recommendation_framework.csv`
- `company_writeups.csv`
- `data_quality.csv`
- `run_health.csv` / `run_health.json`
- `section_health.csv`
- `sheet_health.csv`
- `chart_health.csv`
- `ranking_health.csv`
- `fetch_attempts.csv`
- `provider_usage.csv`
- `oil_news_event_days.csv`
- `oil_news_articles.csv`
- `oil_news_catalysts.csv`
- `oil_news_episode_summary.csv`
- `event_episodes.csv`
- `event_episode_articles.csv`
- `regime_state.csv`
- `regime_history.csv`
- `market_constraints.csv`
- `market_constraints_methodology.csv`
- `confidence_framework.csv`
- `confidence_audit.csv`
- `peer_baskets.csv`
- `peer_basket_membership.csv`
- `company_case_packet_index.csv`
- `returns_daily_matrix.csv`
- `returns_weekly_matrix.csv`
- `cumulative_returns_matrix.csv`
- `benchmark_returns_matrix.csv`
- `correlation_matrix.csv`
- `rolling_correlations.csv`
- `regression_coefficients.csv`
- `regression_diagnostics.csv`
- `rolling_betas.csv`
- `event_study_summary.csv`
- `event_study_detail.csv`
- `cluster_assignments.csv`
- `metric_zscores.csv`
- `metric_percentiles.csv`
- `assumptions_registry.csv`
- `source_log.csv`
- `missing_data_log.csv`
- `analyst_overrides.csv`

## Data Quality / Transparency

- Run-health gate prevents polished publishable conclusions when core market-data backbone is invalid.
- Every estimate is tracked in assumptions registry.
- Missing fields and failed source attempts are logged.
- Source log includes field-level source URLs and source tiers.
- Exact vs estimated/proxy labels are preserved in outputs.
- Confidence penalties reduce ranking scores for low-confidence data.
- Packet confidence is explicitly aligned to ranking health and publishability, with `downstream_readiness_flag` for Repo 2/3 gating.
- Market constraints are labeled heuristic overlays for downstream scenario work, not final valuation assumptions.

## Testing

```powershell
python -m pytest
```

## Known Limitations

- Free public sources cannot provide fully harmonized global station-level weekly retail prices.
- Segment disclosures vary across companies; normalized frameworks remain necessary.
- Some forward event data (e.g., full catalyst schedules) can be partially estimated in free mode.

## Premium API Extension Points

Modules designed for future premium upgrades:

- `src/company_profiles.py`
- `src/hormuz_exposure.py`
- `src/fuel_prices.py`
- `src/operating_mix.py`
- `src/valuation.py`
- `src/catalyst_calendar.py`
