"""Tests for `fava` package."""

from fava.geometry import GEOMETRY


def test_geometry():

    for g in ("cartesian", "polar", "cylindrical", "spherical", "polar"):
        assert GEOMETRY(g).name == g.upper()
    
