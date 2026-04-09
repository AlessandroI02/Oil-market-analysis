from __future__ import annotations

import logging

import pandas as pd

from src.models import RouteRiskRecord

logger = logging.getLogger(__name__)


def _get_weight(weight_map: dict[str, float], key: str) -> float:
    return float(weight_map.get(key, 0.0))


def build_route_risks(profiles_df: pd.DataFrame) -> pd.DataFrame:
    records: list[RouteRiskRecord] = []

    for _, row in profiles_df.iterrows():
        prod = row["production_region_weights"]
        dest = row["destination_market_weights"]
        ref = row["refinery_region_weights"]

        middle_east_presence = _get_weight(prod, "Middle East") + _get_weight(ref, "Middle East")
        asia_dest = _get_weight(dest, "Asia")
        europe_dest = _get_weight(dest, "Europe")
        africa_prod = _get_weight(prod, "Africa")
        na_prod = _get_weight(prod, "North America")

        hormuz_used = middle_east_presence > 0.15
        chokepoints: list[str] = []
        disruption_notes: list[str] = []

        if hormuz_used:
            chokepoints.append("Strait of Hormuz")
            disruption_notes.append("Material reliance on Gulf-origin crude/feedstock transit through Hormuz.")

        if hormuz_used and (asia_dest > 0.2 or europe_dest > 0.2):
            chokepoints.append("Bab el-Mandeb")
            disruption_notes.append("Southbound Gulf cargoes may transit Red Sea approach, exposed to Bab el-Mandeb disruptions.")

        if europe_dest > 0.2:
            chokepoints.append("Suez Canal")
            disruption_notes.append("Europe-linked product and crude trade increases sensitivity to Suez/Red Sea constraints.")

        if africa_prod > 0.2:
            disruption_notes.append("Elevated jurisdiction and offshore logistics risks in African production hubs.")

        if _get_weight(prod, "North America") > 0.6 and asia_dest > 0.2:
            disruption_notes.append("Pacific/Atlantic reroute optionality lowers Hormuz reliance but raises freight duration risk.")

        if middle_east_presence > 0.25:
            disruption_notes.append("Potential sanctions and regional security risk can compound route disruptions.")

        if not chokepoints:
            chokepoints = ["Limited chokepoint concentration"]

        unique_chokepoints = sorted(set(chokepoints))

        hormuz_share = min(max(middle_east_presence * 55, 0.0), 85.0)
        bab_share = min(max((0.45 if hormuz_used else 0.1) * (asia_dest + europe_dest) * 100, 0.0), 45.0)
        suez_share = min(max(europe_dest * 30, 0.0), 35.0)
        non_chokepoint = max(100 - (hormuz_share + bab_share + suez_share), 0.0)

        pipeline_bypass = "High" if na_prod > 0.55 else ("Medium" if na_prod > 0.25 else "Low")
        destination_concentration = max(dest.values()) if dest else 1.0
        rerouting_flex = "High" if destination_concentration < 0.45 else ("Medium" if destination_concentration < 0.65 else "Low")

        risk_level = "Low"
        if hormuz_used and len(unique_chokepoints) >= 3:
            risk_level = "High"
        elif hormuz_used or len(unique_chokepoints) >= 2:
            risk_level = "Medium"

        route_summary = (
            f"Primary hydrocarbon movement links production regions {', '.join(prod.keys())} "
            f"to destination markets {', '.join(dest.keys())}."
        )

        record = RouteRiskRecord(
            company_name=row["company_name"],
            ticker=row["ticker"],
            likely_route_summary=route_summary,
            hormuz_used=hormuz_used,
            hormuz_share_pct=round(hormuz_share, 2),
            bab_el_mandeb_share_pct=round(bab_share, 2),
            suez_share_pct=round(suez_share, 2),
            non_chokepoint_share_pct=round(non_chokepoint, 2),
            pipeline_bypass_optionality=pipeline_bypass,
            rerouting_flexibility=rerouting_flex,
            other_chokepoints=[cp for cp in unique_chokepoints if cp != "Strait of Hormuz"],
            disruption_notes=" ".join(disruption_notes) if disruption_notes else "No major disruption note generated.",
            qualitative_route_risk=risk_level,
        )
        records.append(record)

    out = pd.DataFrame([r.model_dump() for r in records])
    out["other_chokepoints"] = out["other_chokepoints"].apply(lambda x: " | ".join(x))
    logger.info("Generated route risk records for %s companies", len(out))
    return out
