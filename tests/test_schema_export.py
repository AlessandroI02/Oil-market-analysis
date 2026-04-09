from pathlib import Path

from src.schema_export import export_handoff_schemas


def test_schema_export_writes_required_files():
    out = export_handoff_schemas(Path("."))
    assert set(out.keys()) == {
        "company_case_packet_schema",
        "market_constraints_schema",
        "regime_state_schema",
        "event_episode_schema",
    }
    for path in out.values():
        assert path.exists()
