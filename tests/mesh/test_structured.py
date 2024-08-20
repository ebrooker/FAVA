"""Tests for `fava` package."""

import pytest
from fava.mesh.structured import Structured

@pytest.fixture
def my_mesh():
    return Structured()

def test_init(my_mesh):
    assert my_mesh.mesh_type == "Structured"

def test_is_this_your_method():
    assert not Structured.is_this_your_mesh()