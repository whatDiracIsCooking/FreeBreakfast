"""tools/basis/cartesian_to_spherical.py: what it relabels, and what it must not.

It is the step between a Cartesian BSE download and a file src/io/basis will
parse, so these cases pin that it changes only harmonic tags -- never an
exponent or coefficient -- and that a second run is a no-op, which is what
keeps data/basis/ reproducible byte for byte.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_SCRIPT = _TOOLS / "basis" / "cartesian_to_spherical.py"
_spec = importlib.util.spec_from_file_location("cartesian_to_spherical", _SCRIPT)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def shell(function_type: str, *ls: int) -> dict:
    return {
        "function_type": function_type,
        "region": "",
        "angular_momentum": list(ls),
        "exponents": ["1.5", "0.25"],
        "coefficients": [["0.4", "0.7"] for _ in ls],
    }


def document() -> dict:
    """6-31G's potassium, cut down: bare-gto s and sp, Cartesian d (and an f)."""
    return {
        "revision_description": "Data from Gaussian 09/GAMESS",
        "function_types": ["gto", "gto_cartesian"],
        "name": "6-31G",
        "elements": {
            "1": {"electron_shells": [shell("gto", 0)]},
            "19": {
                "electron_shells": [
                    shell("gto", 0),
                    shell("gto", 0, 1),
                    shell("gto_cartesian", 2),
                    shell("gto_cartesian", 3),
                ]
            },
        },
    }


def test_relabels_cartesian_shells_and_nothing_else() -> None:
    before = document()
    after = copy.deepcopy(before)
    assert tool.to_spherical(after) == 2

    k_before = before["elements"]["19"]["electron_shells"]
    k_after = after["elements"]["19"]["electron_shells"]
    assert [s["function_type"] for s in k_after] == [
        "gto",
        "gto",
        "gto_spherical",
        "gto_spherical",
    ]
    for old, new in zip(k_before, k_after, strict=True):
        assert new["exponents"] == old["exponents"]
        assert new["coefficients"] == old["coefficients"]
        assert new["angular_momentum"] == old["angular_momentum"]
    assert after["elements"]["1"] == before["elements"]["1"]
    assert after["name"] == "6-31G"


def test_header_records_what_was_done() -> None:
    doc = document()
    tool.to_spherical(doc)
    assert doc["function_types"] == ["gto", "gto_spherical"]
    assert doc["revision_description"] == f"Data from Gaussian 09/GAMESS; {tool.NOTE}"


def test_a_spherical_document_is_left_alone() -> None:
    doc = document()
    tool.to_spherical(doc)
    again = copy.deepcopy(doc)
    assert tool.to_spherical(again) == 0
    assert again == doc


def test_cli_rewrites_in_place_and_twice_is_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "6-31g.json"
    path.write_text(json.dumps(document(), indent=2) + "\n", encoding="utf-8")
    assert tool.main([str(path)]) == 0
    first = path.read_bytes()
    assert b"gto_cartesian" not in first
    assert tool.main([str(path)]) == 0
    assert path.read_bytes() == first


def test_cli_rejects_a_document_without_elements(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert tool.main([str(path)]) == 1
