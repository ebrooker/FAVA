"""Tests for `fava` package."""

import pytest
from fava.mesh.unstructured import Unstructured

@pytest.fixture
def my_mesh():
    return Unstructured()

def test_init(my_mesh):
    assert my_mesh.mesh_type == "Unstructured"

def test_is_this_your_method():
    assert not Unstructured.is_this_your_mesh()