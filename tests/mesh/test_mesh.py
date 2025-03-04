"""Tests for `fava` package."""

import pytest

from fava.mesh import Mesh


@pytest.fixture
def my_mesh():
    return Mesh()


def test_init(my_mesh):
    assert my_mesh.mesh_type == "Mesh"


def test_is_this_your_method():
    assert not Mesh.is_this_your_mesh()
