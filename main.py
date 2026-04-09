from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.assumptions_registry import AssumptionsRegistry, MissingDataLogger
from src.analyst_overrides import (
    apply_profile_overrides,
    apply_universe_overrides,
    build_overrides_log,
    load_overrides,
)
from src.archetypes import build_archetypes
from src.catalyst_calendar import build_catalyst_calendar, split_catalyst_calendar
from src.charts import create_charts
from src.cli import parse_args
from src.company_case_packets import build_company_case_packets
from src.company_profiles import build_company_profiles
from src.company_writeups import build_company_writeups
from src.confidence_framework import build_confidence_audit, build_confidence_framework
from src.config import RuntimeConfig, ensure_directories, load_settings, timestamp_label
from src.data_quality import build_data_quality_table
from src.earnings_calendar import build_earnings_calendar
from src.equity_prices import build_equity_tracker
from src.excel_export import export_excel_workbook
from src.factor_decomposition import build_factor_decomposition
from src.fetch_diagnostics import FetchDiagnostics
from src.fuel_prices import build_fuel_trackers
from src.health_reporting import (
    build_chart_health,
    build_ranking_health,
    build_section_health,
    build_sheet_health,
)
from src.historical_analogues import build_historical_analogues
from src.hormuz_exposure import build_route_exposure_build, estimate_hormuz_exposure
from src.logging_config import configure_logging
from src.market_constraints import build_market_constraints, build_market_constraints_methodology
from src.market_math import build_market_math_exports
from src.market_prices import build_crude_tracker
from src.oil_news_report import build_oil_news_report
from src.operating_mix import build_operating_mix
from src.peer_baskets import build_peer_baskets
from src.regime_engine import build_regime_state
from src.report_writer import build_insights
from src.ranking_framework import build_rankings
from src.route_risk import build_route_risks
from src.run_health import assess_run_health, export_run_health
from src.scenario_analysis import build_scenario_analysis
from src.schema_export import export_handoff_schemas
from src.source_logger import SourceLogger
from src.storage_paths import ensure_storage_layout
from src.universe_builder import build_universe
from src.universe_review import export_universe_review
from src.utils.dates import derive_date_window
from src.utils.yfinance_utils import configure_yfinance_cache
from src.valuation import build_valuation_table
from src.word_export import export_word_thesis


def _export_debug_csvs(debug_dir: Path, datasets: dict[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename, frame in datasets.items():
        path = debug_dir / filename
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(path, index=False)
        else:
            pd.DataFrame([{"value": str(frame)}]).to_csv(path, index=False)
        paths[filename] = path
    return paths


def run() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    settings_path = root_dir / "config" / "settings.yaml"
    overrides_path = root_dir / "config" / "analyst_overrides.yaml"
    scenario_settings_path = root_dir / "config" / "scenario_settings.yaml"
    settings = load_settings(settings_path)
    overrides = load_overrides(overrides_path)
    overrides_log_df = build_overrides_log(overrides)

    lookback_months = args.lookback_months or settings.run.lookback_months
    frequency = args.frequency or settings.run.frequency
    start_date, end_date = derive_date_window(args.start_date, args.end_date, lookback_months)

    runtime = RuntimeConfig(
        start_date=start_date,
        end_date=end_date,
        lookback_months=lookback_months,
        frequency=frequency,
        output_dir=args.output_dir,
        max_companies=args.max_companies or settings.run.max_companies,
        rebuild_cache=args.rebuild_cache or settings.run.rebuild_cache,
        skip_word=args.skip_word,
        skip_excel=args.skip_excel,
        only_universe=args.only_universe,
        only_review=args.only_review,
        debug=args.debug or settings.run.debug,
    )

    paths = ensure_directories(root_dir, settings, output_override=runtime.output_dir)
    storage_paths = ensure_storage_layout(root_dir)
    configure_logging(paths["logs"], debug=runtime.debug)
    logger = logging.getLogger("pipeline")

    configure_yfinance_cache(paths["cache"] / "yfinance")

    ts_label = timestamp_label()
    logger.info("Starting pipeline | start=%s end=%s freq=%s", runtime.start_date, runtime.end_date, runtime.frequency)

    source_logger = SourceLogger()
    assumptions_registry = AssumptionsRegistry()
    missing_logger = MissingDataLogger()
    fetch_diagnostics = FetchDiagnostics()

    # Phase 2: Universe review pipeline
    universe = build_universe(
        include_secondary=settings.universe.include_secondary_bucket,
        min_market_cap_usd=settings.universe.min_market_cap_usd,
        source_logger=source_logger,
        max_companies=runtime.max_companies,
    )
    reviewed_df = apply_universe_overrides(universe.reviewed, overrides)
    included_df = reviewed_df[reviewed_df["bucket_classification"].isin(["primary", "secondary"])].copy()
    rejected_df = reviewed_df[reviewed_df["bucket_classification"] == "rejected"].copy()
    if runtime.max_companies is not None:
        included_df = included_df.head(runtime.max_companies)

    review_paths = export_universe_review(
        reviewed_df=reviewed_df,
        included_df=included_df,
        rejected_df=rejected_df,
        debug_dir=paths["output_debug"],
    )

    logger.info("Universe review artifacts saved: %s", review_paths)

    source_log_path = paths["output_debug"] / "source_log.csv"
    assumptions_path = paths["output_debug"] / "assumptions_registry.csv"
    missing_path = paths["output_debug"] / "missing_data_log.csv"
    overrides_path_debug = paths["output_debug"] / "analyst_overrides.csv"

    source_logger.export_csv(source_log_path)
    assumptions_registry.export_csv(assumptions_path)
    missing_logger.export_csv(missing_path)
    overrides_log_df.to_csv(overrides_path_debug, index=False)

    if runtime.only_universe or runtime.only_review:
        logger.info("Run completed in universe-only mode.")
        return 0

    # Phase 3+: data ingestion and estimation modules
    profiles_df = build_company_profiles(
        included_df,
        assumptions_registry=assumptions_registry,
        source_logger=source_logger,
        missing_logger=missing_logger,
    )
    profiles_df = apply_profile_overrides(profiles_df, overrides)

    exposure_df = estimate_hormuz_exposure(
        profiles_df,
        assumptions_registry=assumptions_registry,
    )
    route_exposure_build_df = build_route_exposure_build(profiles_df, exposure_df)
    route_risks_df = build_route_risks(profiles_df)
    chokepoint_exposure_df = route_risks_df[
        [
            "company_name",
            "ticker",
            "hormuz_share_pct",
            "bab_el_mandeb_share_pct",
            "suez_share_pct",
            "non_chokepoint_share_pct",
            "pipeline_bypass_optionality",
            "rerouting_flexibility",
            "qualitative_route_risk",
        ]
    ].copy()

    crude_tracker_df = build_crude_tracker(
        runtime.start_date,
        runtime.end_date,
        runtime.frequency,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )

    fuel_tracker_df, fuel_weights_df = build_fuel_trackers(
        profiles_df=profiles_df,
        crude_tracker_df=crude_tracker_df,
        start_date=runtime.start_date,
        end_date=runtime.end_date,
        frequency=runtime.frequency,
        source_logger=source_logger,
        assumptions_registry=assumptions_registry,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )

    equity_tracker_df = build_equity_tracker(
        included_df=included_df,
        crude_tracker_df=crude_tracker_df,
        fuel_tracker_df=fuel_tracker_df,
        start_date=runtime.start_date,
        end_date=runtime.end_date,
        frequency=runtime.frequency,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )

    operating_mix_df = build_operating_mix(
        profiles_df,
        source_logger=source_logger,
        assumptions_registry=assumptions_registry,
        missing_logger=missing_logger,
    )

    earnings_df = build_earnings_calendar(
        included_df,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )
    archetypes_df = build_archetypes(included_df, operating_mix_df)

    valuation_df = build_valuation_table(
        included_df=included_df,
        source_logger=source_logger,
        assumptions_registry=assumptions_registry,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )
    scenario_df, scenario_event_path_df = build_scenario_analysis(
        exposure_df=exposure_df,
        operating_mix_df=operating_mix_df,
        valuation_df=valuation_df,
        crude_tracker_df=crude_tracker_df,
        brent_levels=settings.v2.scenario_brent_levels,
    )
    factor_df = build_factor_decomposition(
        equity_tracker_df=equity_tracker_df,
        crude_tracker_df=crude_tracker_df,
        start_date=runtime.start_date,
        end_date=runtime.end_date,
        frequency=runtime.frequency,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )
    historical_analogues_df, historical_analogue_company_df = build_historical_analogues(
        included_df=included_df,
        config_path=scenario_settings_path,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )
    catalyst_calendar_df = build_catalyst_calendar(
        included_df=included_df,
        earnings_df=earnings_df,
        config_path=scenario_settings_path,
        source_logger=source_logger,
        missing_logger=missing_logger,
    )
    catalyst_primary_df, catalyst_archive_df = split_catalyst_calendar(
        catalyst_df=catalyst_calendar_df,
        as_of_date=runtime.end_date,
        horizon_days=180,
    )

    oil_news_output_path = paths["output_word"] / f"oil_market_news_1m_{ts_label}.docx"
    oil_news_pkg = build_oil_news_report(
        output_word_path=oil_news_output_path,
        debug_dir=paths["output_debug"],
        threshold_pct=float(settings.quality.news_price_move_threshold_pct),
        lookback_days=int(settings.quality.news_lookback_days),
        max_articles_per_event_day=int(settings.quality.max_articles_per_event_day),
        generate_word=not runtime.skip_word,
        source_logger=source_logger,
        missing_logger=missing_logger,
        fetch_diagnostics=fetch_diagnostics,
    )

    market_math = build_market_math_exports(
        included_df=included_df,
        exposure_df=exposure_df,
        route_risks_df=route_risks_df,
        event_days_df=oil_news_pkg.get("events_df", pd.DataFrame()),
        start_date=runtime.start_date,
        end_date=runtime.end_date,
        raw_market_dir=storage_paths["data_raw_market"],
        interim_market_dir=storage_paths["data_interim_market"],
    )
    regression_summary_df = market_math.get("regression_coefficients", pd.DataFrame())
    rolling_beta_df = market_math.get("rolling_betas", pd.DataFrame())
    rolling_beta_summary_df = (
        rolling_beta_df.sort_values("date").groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
        if not rolling_beta_df.empty and {"date", "ticker"}.issubset(set(rolling_beta_df.columns))
        else pd.DataFrame()
    )
    event_study_summary_df = market_math.get("event_study_summary", pd.DataFrame())
    event_study_detail_df = market_math.get("event_study_detail", pd.DataFrame())
    cluster_assignments_df = market_math.get("cluster_assignments", pd.DataFrame())
    metric_zscores_df = market_math.get("metric_zscores", pd.DataFrame())
    metric_percentiles_df = market_math.get("metric_percentiles", pd.DataFrame())
    returns_daily_matrix_df = market_math.get("returns_daily_matrix", pd.DataFrame())
    returns_weekly_matrix_df = market_math.get("returns_weekly_matrix", pd.DataFrame())
    cumulative_returns_matrix_df = market_math.get("cumulative_returns_matrix", pd.DataFrame())
    benchmark_returns_matrix_df = market_math.get("benchmark_returns_matrix", pd.DataFrame())
    correlation_matrix_df = market_math.get("correlation_matrix", pd.DataFrame())
    rolling_correlations_df = market_math.get("rolling_correlations", pd.DataFrame())

    regime_state_df, regime_history_df = build_regime_state(
        crude_tracker_df=crude_tracker_df,
        benchmark_returns_df=benchmark_returns_matrix_df,
        route_risks_df=route_risks_df,
        event_episodes_df=oil_news_pkg.get("episodes_df", pd.DataFrame()),
        as_of_date=runtime.end_date,
    )

    source_log_df = source_logger.to_dataframe()
    assumptions_df = assumptions_registry.to_dataframe()
    missing_df = missing_logger.to_dataframe()
    data_quality_df = build_data_quality_table(
        included_df=included_df,
        assumptions_df=assumptions_df,
        missing_df=missing_df,
        source_log_df=source_log_df,
    )
    peer_baskets_df, peer_basket_membership_df = build_peer_baskets(
        included_df=included_df,
        archetypes_df=archetypes_df,
        route_risks_df=route_risks_df,
        factor_df=factor_df,
        exposure_df=exposure_df,
    )

    rankings = build_rankings(
        included_df=included_df,
        archetypes_df=archetypes_df,
        exposure_df=exposure_df,
        route_risks_df=route_risks_df,
        valuation_df=valuation_df,
        scenario_df=scenario_df,
        event_path_df=scenario_event_path_df,
        factor_df=factor_df,
        catalyst_df=catalyst_primary_df,
        data_quality_df=data_quality_df,
        weight_config=settings.v2.ranking_weights.model_dump(),
        confidence_penalty_map=settings.v2.confidence_penalty,
    )
    core_ranking_df = rankings["core_ranking"]
    extended_ranking_df = rankings["extended_ranking"]
    recommendation_framework_df = rankings["recommendation_framework"]
    company_scorecards_df = rankings["company_scorecards"]

    ranking_health_df = build_ranking_health(core_ranking_df, extended_ranking_df)
    confidence_framework_df = build_confidence_framework(
        included_df=included_df,
        data_quality_df=data_quality_df,
        source_log_df=source_log_df,
        route_risks_df=route_risks_df,
        event_study_summary_df=event_study_summary_df,
        regime_state_df=regime_state_df,
        ranking_health_df=ranking_health_df,
    )
    confidence_audit_df = build_confidence_audit(
        confidence_framework_df=confidence_framework_df,
        ranking_health_df=ranking_health_df,
    )
    market_constraints_df = build_market_constraints(
        included_df=included_df,
        exposure_df=exposure_df,
        route_risks_df=route_risks_df,
        factor_df=factor_df,
        scenario_df=scenario_df,
        data_quality_df=data_quality_df,
        regime_state_df=regime_state_df,
        as_of_date=runtime.end_date,
        confidence_framework_df=confidence_framework_df,
        event_study_summary_df=event_study_summary_df,
    )
    market_constraints_methodology_df = build_market_constraints_methodology(as_of_date=runtime.end_date)

    company_writeups_df = build_company_writeups(
        core_ranking_df=core_ranking_df,
        extended_ranking_df=extended_ranking_df,
        recommendation_df=recommendation_framework_df,
    )
    company_case_packet_index_df = build_company_case_packets(
        included_df=included_df,
        archetypes_df=archetypes_df,
        exposure_df=exposure_df,
        route_risks_df=route_risks_df,
        crude_tracker_df=crude_tracker_df,
        fuel_tracker_df=fuel_tracker_df,
        equity_tracker_df=equity_tracker_df,
        factor_df=factor_df,
        regression_coefficients_df=regression_summary_df,
        rolling_betas_df=rolling_beta_df,
        event_study_summary_df=event_study_summary_df,
        historical_analogues_df=historical_analogues_df,
        catalyst_primary_df=catalyst_primary_df,
        market_constraints_df=market_constraints_df,
        confidence_framework_df=confidence_framework_df,
        regime_state_df=regime_state_df,
        source_log_df=source_log_df,
        output_dir=storage_paths["output_packets"],
        as_of_date=runtime.end_date,
        root_dir=root_dir,
    )
    schema_paths = export_handoff_schemas(root_dir)

    # Chart package + chart health
    chart_sheets = create_charts(
        exposure_df=exposure_df,
        core_ranking_df=core_ranking_df,
        extended_ranking_df=extended_ranking_df,
        scenario_df=scenario_df,
        valuation_df=valuation_df,
        catalyst_df=catalyst_primary_df,
        crude_tracker_df=crude_tracker_df,
        fuel_tracker_df=fuel_tracker_df,
        equity_tracker_df=equity_tracker_df,
        operating_mix_df=operating_mix_df,
        route_risks_df=route_risks_df,
        earnings_df=earnings_df,
        chart_dir=paths["output_charts"],
        min_non_null_ratio=settings.quality.minimum_non_null_ratio_for_chart,
        min_points=settings.quality.minimum_required_price_points,
    )
    chart_health_df = build_chart_health(chart_sheets)

    # Core datasets dictionary
    datasets = {
        "Universe": included_df,
        "Universe_Review": reviewed_df,
        "Archetypes": archetypes_df,
        "Regime_State": regime_state_df,
        "Regime_History": regime_history_df,
        "Market_Constraints": market_constraints_df,
        "Market_Constraints_Methodology": market_constraints_methodology_df,
        "Event_Episodes": oil_news_pkg.get("episodes_df", pd.DataFrame()),
        "Event_Episode_Articles": oil_news_pkg.get("episode_articles_df", pd.DataFrame()),
        "Confidence_Framework": confidence_framework_df,
        "Confidence_Audit": confidence_audit_df,
        "Peer_Baskets": peer_baskets_df,
        "Peer_Basket_Membership": peer_basket_membership_df,
        "Company_Case_Packet_Index": company_case_packet_index_df,
        "Regression_Summary": regression_summary_df,
        "Rolling_Beta_Summary": rolling_beta_summary_df,
        "Event_Study_Summary": event_study_summary_df,
        "Core_Ranking": core_ranking_df,
        "Extended_Ranking": extended_ranking_df,
        "Route_Exposure_Build": route_exposure_build_df,
        "Chokepoint_Exposure": chokepoint_exposure_df,
        "Hormuz_Ranking": exposure_df,
        "Route_Risks": route_risks_df,
        "Crude_Tracker": crude_tracker_df,
        "Fuel_Tracker": fuel_tracker_df,
        "Fuel_Geography_Weights": fuel_weights_df,
        "Equity_Tracker": equity_tracker_df,
        "Factor_Decomposition": factor_df,
        "Operating_Mix": operating_mix_df,
        "Valuation": valuation_df,
        "Scenario_Analysis": scenario_df,
        "Scenario_Event_Path": scenario_event_path_df,
        "Historical_Analogues": historical_analogues_df,
        "Historical_Analogue_Company": historical_analogue_company_df,
        "Catalyst_Calendar": catalyst_calendar_df,
        "Catalyst_Calendar_Primary": catalyst_primary_df,
        "Catalyst_Calendar_Archive": catalyst_archive_df,
        "Earnings": earnings_df,
        "Analyst_Overrides": overrides_log_df,
        "Recommendation_Framework": recommendation_framework_df,
        "Company_Scorecards": company_scorecards_df,
        "Company_Writeups": company_writeups_df,
        "Data_Quality": data_quality_df,
        "Assumptions": assumptions_df,
        "Source_Log": source_log_df,
        "Missing_Data_Log": missing_df,
    }

    # Run-health and section/sheet health
    run_summary, run_health_df = assess_run_health(
        datasets=datasets,
        quality_cfg=settings.quality.model_dump(),
    )
    section_health_df = build_section_health(
        datasets=datasets,
        run_summary=run_summary,
        min_non_null_ratio=settings.quality.minimum_non_null_ratio_for_section,
    )

    # Include section and ranking health into datasets before sheet-health evaluation
    datasets["Section_Health"] = section_health_df
    datasets["Ranking_Health"] = ranking_health_df
    datasets["Run_Health"] = run_health_df
    datasets["Chart_Health"] = chart_health_df

    sheet_health_df = build_sheet_health(
        datasets={k: v for k, v in datasets.items() if isinstance(v, pd.DataFrame)},
        min_non_null_ratio=settings.quality.minimum_non_null_ratio_for_sheet,
    )
    datasets["Sheet_Health"] = sheet_health_df

    # Phase 4: processed datasets & debug exports
    debug_artifacts = _export_debug_csvs(
        paths["output_debug"],
        {
            "hormuz_exposure_estimates.csv": exposure_df,
            "route_exposure_build.csv": route_exposure_build_df,
            "chokepoint_exposure.csv": chokepoint_exposure_df,
            "route_risks.csv": route_risks_df,
            "regime_state.csv": regime_state_df,
            "regime_history.csv": regime_history_df,
            "market_constraints.csv": market_constraints_df,
            "market_constraints_methodology.csv": market_constraints_methodology_df,
            "event_episodes.csv": oil_news_pkg.get("episodes_df", pd.DataFrame()),
            "event_episode_articles.csv": oil_news_pkg.get("episode_articles_df", pd.DataFrame()),
            "fuel_geography_weights.csv": fuel_weights_df,
            "crude_tracker.csv": crude_tracker_df,
            "fuel_tracker.csv": fuel_tracker_df,
            "equity_tracker.csv": equity_tracker_df,
            "operating_mix.csv": operating_mix_df,
            "archetypes.csv": archetypes_df,
            "valuation.csv": valuation_df,
            "scenario_analysis.csv": scenario_df,
            "scenario_event_path.csv": scenario_event_path_df,
            "factor_decomposition.csv": factor_df,
            "historical_analogues.csv": historical_analogues_df,
            "historical_analogue_company.csv": historical_analogue_company_df,
            "catalyst_calendar.csv": catalyst_calendar_df,
            "catalyst_calendar_primary.csv": catalyst_primary_df,
            "catalyst_calendar_archive.csv": catalyst_archive_df,
            "confidence_framework.csv": confidence_framework_df,
            "confidence_audit.csv": confidence_audit_df,
            "peer_baskets.csv": peer_baskets_df,
            "peer_basket_membership.csv": peer_basket_membership_df,
            "company_case_packet_index.csv": company_case_packet_index_df,
            "regression_coefficients.csv": regression_summary_df,
            "regression_diagnostics.csv": market_math.get("regression_diagnostics", pd.DataFrame()),
            "rolling_betas.csv": rolling_beta_df,
            "event_study_summary.csv": event_study_summary_df,
            "event_study_detail.csv": event_study_detail_df,
            "cluster_assignments.csv": cluster_assignments_df,
            "metric_zscores.csv": metric_zscores_df,
            "metric_percentiles.csv": metric_percentiles_df,
            "returns_daily_matrix.csv": returns_daily_matrix_df,
            "returns_weekly_matrix.csv": returns_weekly_matrix_df,
            "cumulative_returns_matrix.csv": cumulative_returns_matrix_df,
            "benchmark_returns_matrix.csv": benchmark_returns_matrix_df,
            "correlation_matrix.csv": correlation_matrix_df,
            "rolling_correlations.csv": rolling_correlations_df,
            "core_ranking.csv": core_ranking_df,
            "extended_ranking.csv": extended_ranking_df,
            "recommendation_framework.csv": recommendation_framework_df,
            "company_scorecards.csv": company_scorecards_df,
            "company_writeups.csv": company_writeups_df,
            "earnings_dates.csv": earnings_df,
            "source_log.csv": source_log_df,
            "assumptions_registry.csv": assumptions_df,
            "missing_data_log.csv": missing_df,
            "data_quality.csv": data_quality_df,
            "analyst_overrides.csv": overrides_log_df,
            "section_health.csv": section_health_df,
            "sheet_health.csv": sheet_health_df,
            "chart_health.csv": chart_health_df,
            "ranking_health.csv": ranking_health_df,
        },
    )

    run_health_paths = export_run_health(paths["output_debug"], run_summary, run_health_df)
    fetch_diag_paths = fetch_diagnostics.export(paths["output_debug"])
    debug_artifacts.update(run_health_paths)
    debug_artifacts.update(fetch_diag_paths)
    debug_artifacts.update(
        {
            "oil_news_event_days": oil_news_pkg.get("oil_news_event_days"),
            "oil_news_articles": oil_news_pkg.get("oil_news_articles"),
            "oil_news_catalysts": oil_news_pkg.get("oil_news_catalysts"),
            "oil_news_episode_summary": oil_news_pkg.get("oil_news_episode_summary"),
            "event_episodes": oil_news_pkg.get("event_episodes"),
            "event_episode_articles": oil_news_pkg.get("event_episode_articles"),
        }
    )
    debug_artifacts.update(schema_paths)

    logger.info("Debug artifacts saved: %s", debug_artifacts)

    metadata = {
        "model_version": settings.v2.model_version,
        "assumptions_version": settings.v2.assumptions_version,
        "data_cut_date": runtime.end_date.isoformat(),
        "source_refresh_date": datetime.now(UTC).date().isoformat(),
        "change_log_summary": "V2.3 final cleanup: confidence/ranking alignment, differentiated heuristic market constraints, stricter catalyst prioritization, improved source-discipline in oil-news episodes, and portable packet indexing.",
    }

    run_status = str(run_summary.get("run_status", "VALID")).upper()
    if run_status == "INVALID":
        logger.error("Run health status is INVALID. %s", run_summary)

    if run_status == "INVALID" and not settings.quality.allow_degraded_run:
        logger.error("Configuration disallows degraded/invalid exports. Stopping before report export.")
        return 2

    # Phase 5: Excel first
    excel_path = None
    if not runtime.skip_excel:
        excel_path = paths["output_excel"] / f"hormuz_integrated_oil_analysis_{ts_label}.xlsx"
        export_excel_workbook(
            output_path=excel_path,
            datasets=datasets,
            chart_sheets=chart_sheets,
            methodology_summary=(
                "V2.3 cleanup framework: auditable run-health gating, confidence alignment to ranking health, heuristic market-constraint overlays, tightened catalyst prioritization, and episode-based news source discipline."
            ),
            metadata=metadata,
            run_summary=run_summary,
            run_health_df=run_health_df,
            sheet_health_df=sheet_health_df,
            chart_health_df=chart_health_df,
        )
        logger.info("Excel workbook exported: %s", excel_path)

    # Phase 6: Word second, consuming processed output
    if not runtime.skip_word:
        insights = build_insights(
            core_ranking_df=core_ranking_df,
            extended_ranking_df=extended_ranking_df,
            valuation_df=valuation_df,
            scenario_df=scenario_df,
            factor_df=factor_df,
            analogue_df=historical_analogues_df,
            catalyst_df=catalyst_primary_df,
            recommendation_df=recommendation_framework_df,
            quality_df=data_quality_df,
            run_summary=run_summary,
            section_health_df=section_health_df,
        )
        word_path = paths["output_word"] / f"hormuz_integrated_oil_thesis_{ts_label}.docx"
        export_word_thesis(
            output_path=word_path,
            datasets=datasets,
            insights=insights,
            chart_sheets=chart_sheets,
            metadata=metadata,
            run_summary=run_summary,
            section_health_df=section_health_df,
        )
        logger.info("Word thesis exported: %s", word_path)
        logger.info("Oil-news Word report exported: %s", oil_news_pkg.get("word_path"))

    if run_status == "INVALID" and settings.quality.fail_on_missing_core_market_data:
        logger.error("Run marked INVALID and fail_on_missing_core_market_data=true. Returning non-zero.")
        return 2

    if run_status == "DEGRADED":
        logger.warning("Pipeline completed in DEGRADED mode. Outputs are qualified and section-suppressed where needed.")
    elif run_status == "INVALID":
        logger.warning("Pipeline completed in INVALID mode with explicit diagnostics package.")
    else:
        logger.info("Pipeline completed successfully with VALID run status.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
