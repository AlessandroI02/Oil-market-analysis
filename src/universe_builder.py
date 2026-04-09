from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from src.models import CompanyUniverseRecord
from src.source_logger import SourceLogger

logger = logging.getLogger(__name__)


@dataclass
class UniverseBuildResult:
    reviewed: pd.DataFrame
    included: pd.DataFrame
    rejected: pd.DataFrame


def _seed_candidates() -> list[dict[str, Any]]:
    return [
        {
            "company_name": "Exxon Mobil",
            "ticker": "XOM",
            "exchange": "NYSE",
            "isin": "US30231G1022",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Global integrated major with upstream, global crude/product sales, and consumer fuel retail network.",
            "source_links": [
                "https://corporate.exxonmobil.com",
                "https://corporate.exxonmobil.com/locations",
                "https://www.sec.gov/ixviewer/ix.html",
            ],
            "short_business_description": "Global integrated oil and gas company with upstream, refining, chemicals, and fuels marketing.",
            "primary_operating_regions": ["North America", "Europe", "Middle East", "Asia Pacific"],
            "confidence": "High",
        },
        {
            "company_name": "Chevron",
            "ticker": "CVX",
            "exchange": "NYSE",
            "isin": "US1667641005",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated supermajor with upstream assets and fuels marketing/retail brands.",
            "source_links": [
                "https://www.chevron.com",
                "https://www.chevron.com/operations/downstream",
                "https://www.sec.gov/ixviewer/ix.html",
            ],
            "short_business_description": "Integrated energy company active in exploration, production, refining, and marketing.",
            "primary_operating_regions": ["North America", "Asia", "Australia", "Middle East"],
            "confidence": "High",
        },
        {
            "company_name": "Shell",
            "ticker": "SHEL",
            "exchange": "NYSE",
            "isin": "GB00BP6MXD84",
            "headquarters_country": "United Kingdom",
            "listing_country": "United Kingdom",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Major integrated producer with large global retail fuels footprint.",
            "source_links": [
                "https://www.shell.com",
                "https://www.shell.com/what-we-do/downstream.html",
                "https://www.shell.com/investors/results-and-reporting.html",
            ],
            "short_business_description": "Integrated oil and gas group spanning upstream, LNG, refining, chemicals, and mobility.",
            "primary_operating_regions": ["Europe", "North America", "Asia Pacific", "Middle East"],
            "confidence": "High",
        },
        {
            "company_name": "BP",
            "ticker": "BP",
            "exchange": "NYSE",
            "isin": "GB0007980591",
            "headquarters_country": "United Kingdom",
            "listing_country": "United Kingdom",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated major with upstream, trading, refining, and retail fuel sales.",
            "source_links": [
                "https://www.bp.com",
                "https://www.bp.com/en/global/corporate/what-we-do/customers-and-products.html",
                "https://www.bp.com/en/global/corporate/investors/results-and-reporting.html",
            ],
            "short_business_description": "Integrated energy major with significant oil and gas production and retail fuels presence.",
            "primary_operating_regions": ["Europe", "United States", "Middle East", "Asia"],
            "confidence": "High",
        },
        {
            "company_name": "TotalEnergies",
            "ticker": "TTE",
            "exchange": "NYSE",
            "isin": "FR0000120271",
            "headquarters_country": "France",
            "listing_country": "France",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated global producer-refiner-marketer with major station network.",
            "source_links": [
                "https://totalenergies.com",
                "https://totalenergies.com/company/activities/refining-chemicals",
                "https://totalenergies.com/investors/publications-and-regulated-information",
            ],
            "short_business_description": "Integrated multi-energy company with oil and gas production and downstream marketing operations.",
            "primary_operating_regions": ["Europe", "Africa", "Middle East", "Asia"],
            "confidence": "High",
        },
        {
            "company_name": "Eni",
            "ticker": "E",
            "exchange": "NYSE",
            "isin": "IT0003132476",
            "headquarters_country": "Italy",
            "listing_country": "Italy",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated major with upstream, refining and branded fuel stations in Europe.",
            "source_links": [
                "https://www.eni.com",
                "https://www.eni.com/en-IT/business/retail-fuel-stations.html",
                "https://www.eni.com/en-IT/investors/results-reports.html",
            ],
            "short_business_description": "Integrated oil and gas company with E&P, refining, marketing and retail operations.",
            "primary_operating_regions": ["Europe", "North Africa", "Sub-Saharan Africa", "Middle East"],
            "confidence": "High",
        },
        {
            "company_name": "Repsol",
            "ticker": "REP.MC",
            "exchange": "BME",
            "isin": "ES0173516115",
            "headquarters_country": "Spain",
            "listing_country": "Spain",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated operator with upstream and broad Iberian retail network.",
            "source_links": [
                "https://www.repsol.com",
                "https://www.repsol.com/en/about-us/what-we-do/index.cshtml",
                "https://www.repsol.com/en/shareholders-and-investors/index.cshtml",
            ],
            "short_business_description": "Integrated European energy company with production, refining, and retail station operations.",
            "primary_operating_regions": ["Europe", "North America", "Latin America"],
            "confidence": "High",
        },
        {
            "company_name": "OMV",
            "ticker": "OMV.VI",
            "exchange": "VSE",
            "isin": "AT0000743059",
            "headquarters_country": "Austria",
            "listing_country": "Austria",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated producer-refiner with Central/Eastern European station footprint.",
            "source_links": [
                "https://www.omv.com",
                "https://www.omv.com/en/about-us/our-business/refining-marketing",
                "https://www.omv.com/en/investors/publications",
            ],
            "short_business_description": "Integrated energy company active in E&P and fuels marketing.",
            "primary_operating_regions": ["Europe", "Middle East", "North Africa"],
            "confidence": "Medium",
        },
        {
            "company_name": "Galp",
            "ticker": "GALP.LS",
            "exchange": "Euronext Lisbon",
            "isin": "PTGAL0AM0009",
            "headquarters_country": "Portugal",
            "listing_country": "Portugal",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated producer and fuel retailer in Iberia with international trading exposure.",
            "source_links": [
                "https://www.galp.com",
                "https://www.galp.com/corp/en/about-us/our-business",
                "https://www.galp.com/corp/en/investors/results-and-presentations",
            ],
            "short_business_description": "Integrated oil and gas company with E&P, refining and retail distribution.",
            "primary_operating_regions": ["Europe", "Brazil", "Africa"],
            "confidence": "Medium",
        },
        {
            "company_name": "Suncor Energy",
            "ticker": "SU",
            "exchange": "NYSE",
            "isin": "CA8672241079",
            "headquarters_country": "Canada",
            "listing_country": "Canada",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated oil sands producer with downstream refining and Petro-Canada retail network.",
            "source_links": [
                "https://www.suncor.com",
                "https://www.petro-canada.ca",
                "https://www.suncor.com/en-ca/investors",
            ],
            "short_business_description": "Integrated Canadian oil producer with refining and retail assets.",
            "primary_operating_regions": ["Canada", "United States"],
            "confidence": "High",
        },
        {
            "company_name": "Orlen",
            "ticker": "PKN.WA",
            "exchange": "WSE",
            "isin": "PLPKN0000018",
            "headquarters_country": "Poland",
            "listing_country": "Poland",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated regional major with upstream participation and extensive station network.",
            "source_links": [
                "https://www.orlen.pl",
                "https://www.orlen.pl/en/about-the-company/our-business",
                "https://www.orlen.pl/en/investors/financial-results",
            ],
            "short_business_description": "Central European integrated energy group active in upstream, refining, and retail.",
            "primary_operating_regions": ["Central Europe", "Baltics"],
            "confidence": "Medium",
        },
        {
            "company_name": "PetroChina",
            "ticker": "0857.HK",
            "exchange": "HKEX",
            "isin": "CNE1000003W8",
            "headquarters_country": "China",
            "listing_country": "China",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Large integrated producer-refiner-marketer with extensive domestic retail network.",
            "source_links": [
                "https://www.petrochina.com.cn",
                "https://www.petrochina.com.cn/ptr/ndbg/",
                "https://www.hkexnews.hk",
            ],
            "short_business_description": "Integrated state-backed oil and gas major with broad downstream assets.",
            "primary_operating_regions": ["China", "Central Asia", "Middle East"],
            "confidence": "Medium",
        },
        {
            "company_name": "Sinopec",
            "ticker": "0386.HK",
            "exchange": "HKEX",
            "isin": "CNE1000002Q2",
            "headquarters_country": "China",
            "listing_country": "China",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Integrated refiner-marketer with upstream assets and large station network.",
            "source_links": [
                "http://www.sinopecgroup.com/group/en",
                "http://www.sinopec.com/listco/en/investor_centre/reportsandpublications/",
                "https://www.hkexnews.hk",
            ],
            "short_business_description": "Integrated Chinese oil major focused on refining, marketing, and upstream production.",
            "primary_operating_regions": ["China", "Middle East", "Africa"],
            "confidence": "Medium",
        },
        {
            "company_name": "Equinor",
            "ticker": "EQNR",
            "exchange": "NYSE",
            "isin": "NO0010096985",
            "headquarters_country": "Norway",
            "listing_country": "Norway",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Near-match: strong upstream exporter but limited direct retail station presence after retail divestments.",
            "source_links": [
                "https://www.equinor.com",
                "https://www.equinor.com/investors/annual-reports",
            ],
            "short_business_description": "Large upstream-focused energy company with international sales footprint.",
            "primary_operating_regions": ["North Sea", "Europe", "North America"],
            "confidence": "High",
        },
        {
            "company_name": "Cenovus Energy",
            "ticker": "CVE",
            "exchange": "NYSE",
            "isin": "CA15135U1093",
            "headquarters_country": "Canada",
            "listing_country": "Canada",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Near-match: integrated upstream-refining profile, but limited branded retail station ownership.",
            "source_links": [
                "https://www.cenovus.com",
                "https://www.cenovus.com/invest/docs/default-source/investor-reports",
            ],
            "short_business_description": "Integrated Canadian producer with upstream and refining operations.",
            "primary_operating_regions": ["Canada", "United States"],
            "confidence": "Medium",
        },
        {
            "company_name": "Saudi Aramco",
            "ticker": "2222.SR",
            "exchange": "Tadawul",
            "isin": "SA14TG012N13",
            "headquarters_country": "Saudi Arabia",
            "listing_country": "Saudi Arabia",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Near-match: dominant upstream exporter with major refining interests but limited direct listed retail network detail.",
            "source_links": [
                "https://www.aramco.com",
                "https://www.aramco.com/en/investors",
            ],
            "short_business_description": "Large integrated national oil company with global export and refining exposure.",
            "primary_operating_regions": ["Middle East", "Asia"],
            "confidence": "Medium",
        },
        {
            "company_name": "Petrobras",
            "ticker": "PBR",
            "exchange": "NYSE",
            "isin": "BRPETRACNPR6",
            "headquarters_country": "Brazil",
            "listing_country": "Brazil",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Near-match: integrated upstream and refining presence, but retail operations largely divested.",
            "source_links": [
                "https://petrobras.com.br",
                "https://petrobras.com.br/en/investors/results-and-publications",
            ],
            "short_business_description": "Integrated Brazilian producer and refiner with strong export exposure.",
            "primary_operating_regions": ["Brazil", "Atlantic Basin"],
            "confidence": "High",
        },
        {
            "company_name": "CNOOC",
            "ticker": "0883.HK",
            "exchange": "HKEX",
            "isin": "HK0883013259",
            "headquarters_country": "China",
            "listing_country": "China",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Near-match: strong upstream and international sales but no material retail station business.",
            "source_links": [
                "https://www.cnoocltd.com",
                "https://www.cnoocltd.com/col/col6551/index.html",
            ],
            "short_business_description": "Upstream-heavy offshore producer with international crude sales.",
            "primary_operating_regions": ["China offshore", "South America", "Africa"],
            "confidence": "High",
        },
        {
            "company_name": "Valero Energy",
            "ticker": "VLO",
            "exchange": "NYSE",
            "isin": "US91913Y1001",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": False,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Reviewed for relevance due to scale, but excluded from integrated universe as predominantly refining/marketing.",
            "source_links": [
                "https://www.valero.com",
                "https://investor.valero.com",
            ],
            "short_business_description": "Large refiner and fuels marketer.",
            "primary_operating_regions": ["United States", "Europe"],
            "confidence": "High",
        },
        {
            "company_name": "Phillips 66",
            "ticker": "PSX",
            "exchange": "NYSE",
            "isin": "US7185461040",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": False,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Reviewed due to scale but excluded from primary integrated definition (no upstream extraction).",
            "source_links": [
                "https://www.phillips66.com",
                "https://investor.phillips66.com",
            ],
            "short_business_description": "Refining, midstream and marketing-focused company.",
            "primary_operating_regions": ["United States"],
            "confidence": "High",
        },
        {
            "company_name": "Marathon Petroleum",
            "ticker": "MPC",
            "exchange": "NYSE",
            "isin": "US56585A1025",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": False,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Reviewed due to market cap but excluded as downstream-focused refiner with no upstream production.",
            "source_links": [
                "https://www.marathonpetroleum.com",
                "https://ir.marathonpetroleum.com",
            ],
            "short_business_description": "Large U.S. refiner and fuel distributor.",
            "primary_operating_regions": ["United States"],
            "confidence": "High",
        },
        {
            "company_name": "Kinder Morgan",
            "ticker": "KMI",
            "exchange": "NYSE",
            "isin": "US49456B1017",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": False,
            "international_sales_exists": False,
            "retail_stations_exists": False,
            "reason_included": "Reviewed as major energy infrastructure name but excluded (midstream only).",
            "source_links": [
                "https://www.kindermorgan.com",
                "https://ir.kindermorgan.com",
            ],
            "short_business_description": "North American midstream pipeline operator.",
            "primary_operating_regions": ["United States"],
            "confidence": "High",
        },
        {
            "company_name": "Enterprise Products Partners",
            "ticker": "EPD",
            "exchange": "NYSE",
            "isin": "US2937921078",
            "headquarters_country": "United States",
            "listing_country": "United States",
            "upstream_exists": False,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Reviewed but excluded under integrated criteria due to midstream concentration.",
            "source_links": [
                "https://www.enterpriseproducts.com",
                "https://ir.enterpriseproducts.com",
            ],
            "short_business_description": "Midstream storage and pipeline operator.",
            "primary_operating_regions": ["United States"],
            "confidence": "High",
        },
    ]


def _discovery_candidates() -> list[dict[str, Any]]:
    """
    Expanded discovery watchlist to widen coverage beyond the original seed set.
    These are still validated by criteria/mkt-cap filters downstream.
    """
    return [
        {
            "company_name": "Ecopetrol",
            "ticker": "EC",
            "exchange": "NYSE",
            "isin": "US2791581091",
            "headquarters_country": "Colombia",
            "listing_country": "Colombia",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Expanded discovery candidate with integrated upstream/downstream profile and listed liquidity.",
            "source_links": [
                "https://www.ecopetrol.com.co",
                "https://www.ecopetrol.com.co/wps/portal/Home/en/investors",
            ],
            "short_business_description": "Listed integrated Latin American energy company with upstream and refining operations.",
            "primary_operating_regions": ["Latin America"],
            "confidence": "Medium",
        },
        {
            "company_name": "MOL Group",
            "ticker": "MOL.BD",
            "exchange": "BSE",
            "isin": "HU0000153937",
            "headquarters_country": "Hungary",
            "listing_country": "Hungary",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Expanded Central European integrated candidate with retail network.",
            "source_links": [
                "https://molgroup.info/en",
                "https://molgroup.info/en/investor-relations",
            ],
            "short_business_description": "Integrated Central European producer-refiner-marketer with station footprint.",
            "primary_operating_regions": ["Central Europe"],
            "confidence": "Medium",
        },
        {
            "company_name": "SK Innovation",
            "ticker": "096770.KS",
            "exchange": "KRX",
            "isin": "KR7096770003",
            "headquarters_country": "South Korea",
            "listing_country": "South Korea",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": False,
            "reason_included": "Expanded Asian near-match integrated candidate.",
            "source_links": [
                "https://eng.skinnovation.com",
                "https://www.skinnovation.com/ir",
            ],
            "short_business_description": "Integrated refining and E&P-linked energy group with global product trade exposure.",
            "primary_operating_regions": ["Asia"],
            "confidence": "Low",
        },
        {
            "company_name": "Indian Oil",
            "ticker": "IOC.NS",
            "exchange": "NSE",
            "isin": "INE242A01010",
            "headquarters_country": "India",
            "listing_country": "India",
            "upstream_exists": False,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Expanded large-cap downstream-heavy near-match with major retail reach.",
            "source_links": [
                "https://iocl.com",
                "https://iocl.com/investor-relations",
            ],
            "short_business_description": "Large listed refining and marketing company with extensive station network.",
            "primary_operating_regions": ["India"],
            "confidence": "Medium",
        },
        {
            "company_name": "Idemitsu Kosan",
            "ticker": "5019.T",
            "exchange": "TSE",
            "isin": "JP3142500002",
            "headquarters_country": "Japan",
            "listing_country": "Japan",
            "upstream_exists": True,
            "international_sales_exists": True,
            "retail_stations_exists": True,
            "reason_included": "Expanded integrated candidate with upstream and retail exposure.",
            "source_links": [
                "https://www.idemitsu.com/en/",
                "https://www.idemitsu.com/en/ir/",
            ],
            "short_business_description": "Integrated Japanese energy company spanning upstream, refining, and retail marketing.",
            "primary_operating_regions": ["Asia"],
            "confidence": "Low",
        },
    ]


def _classify_candidate(
    upstream_exists: bool,
    international_sales_exists: bool,
    retail_stations_exists: bool,
    include_secondary: bool,
) -> tuple[str, Optional[str]]:
    met = [upstream_exists, international_sales_exists, retail_stations_exists]
    score = sum(1 for x in met if x)

    if score == 3:
        return "primary", None
    if include_secondary and score == 2:
        return "secondary", None

    reasons: list[str] = []
    if not upstream_exists:
        reasons.append("No upstream extraction")
    if not international_sales_exists:
        reasons.append("No international hydrocarbon sales")
    if not retail_stations_exists:
        reasons.append("No consumer retail station footprint")
    return "rejected", "; ".join(reasons)


def _fetch_market_cap(ticker: str) -> Optional[float]:
    if yf is None:
        return None
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        market_cap = info.get("market_cap")
        return float(market_cap) if market_cap is not None else None
    except Exception:
        return None


def build_universe(
    include_secondary: bool,
    min_market_cap_usd: int,
    source_logger: SourceLogger,
    max_companies: Optional[int] = None,
) -> UniverseBuildResult:
    records: list[CompanyUniverseRecord] = []
    all_candidates = _seed_candidates() + _discovery_candidates()

    for seed in all_candidates:
        classification, auto_exclusion = _classify_candidate(
            upstream_exists=seed["upstream_exists"],
            international_sales_exists=seed["international_sales_exists"],
            retail_stations_exists=seed["retail_stations_exists"],
            include_secondary=include_secondary,
        )

        market_cap = _fetch_market_cap(seed["ticker"])
        exclusion_reason = auto_exclusion
        notes = seed.get("notes")

        if market_cap is not None and market_cap < min_market_cap_usd:
            classification = "rejected"
            cap_text = f"Market cap {market_cap:,.0f} < threshold {min_market_cap_usd:,.0f}"
            exclusion_reason = f"{exclusion_reason}; {cap_text}" if exclusion_reason else cap_text

        record = CompanyUniverseRecord(
            company_name=seed["company_name"],
            ticker=seed["ticker"],
            exchange=seed["exchange"],
            isin=seed.get("isin"),
            headquarters_country=seed.get("headquarters_country"),
            listing_country=seed.get("listing_country"),
            bucket_candidate=f"{classification}_candidate",
            bucket_classification=classification,
            reason_included=seed.get("reason_included") if classification != "rejected" else None,
            reason_excluded=exclusion_reason,
            upstream_exists=seed["upstream_exists"],
            international_sales_exists=seed["international_sales_exists"],
            retail_stations_exists=seed["retail_stations_exists"],
            source_links=seed.get("source_links", []),
            short_business_description=seed.get("short_business_description"),
            primary_operating_regions=seed.get("primary_operating_regions", []),
            archetype=seed.get("archetype"),
            notes=notes,
            confidence=seed.get("confidence", "Medium"),
            inclusion_confidence=seed.get("confidence", "Medium"),
        )
        records.append(record)

        for src in record.source_links:
            source_logger.add(
                company=record.company_name,
                field="universe_classification",
                source_url=src,
                source_tier="Tier 1",
                evidence_flag="estimated" if classification != "primary" else "proxy",
                comments=f"Universe screening evidence for {record.ticker}",
            )

    reviewed_df = pd.DataFrame([r.model_dump() for r in records])
    reviewed_df["source_links"] = reviewed_df["source_links"].apply(lambda x: " | ".join(x))
    reviewed_df["primary_operating_regions"] = reviewed_df["primary_operating_regions"].apply(
        lambda x: " | ".join(x)
    )

    included_df = reviewed_df[reviewed_df["bucket_classification"].isin(["primary", "secondary"])].copy()
    rejected_df = reviewed_df[reviewed_df["bucket_classification"] == "rejected"].copy()

    included_df = included_df.sort_values(["bucket_classification", "company_name"])
    rejected_df = rejected_df.sort_values("company_name")

    if max_companies is not None:
        included_df = included_df.head(max_companies)

    logger.info(
        "Universe built: %s reviewed, %s included, %s rejected",
        len(reviewed_df),
        len(included_df),
        len(rejected_df),
    )

    return UniverseBuildResult(reviewed=reviewed_df, included=included_df, rejected=rejected_df)
