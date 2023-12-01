"""Tests for `fava` package."""
import pytest
from fava.temporary import Temporary


def test_register_mesh():

    @Temporary.register_mesh()
    class A_mesh:
        def __init__(self, val):
            self.val = val
        def __str__(self):
            return f"A_mesh is {self.val}!"
        
    assert ("A_mesh", A_mesh) in Temporary._Temporary__meshes.items()

def test_register_analysis_callable():

    @Temporary.register_analysis()
    def Z_sum(*args):
        return sum(args)

    assert "Z_sum" in Temporary.__dict__
    assert Temporary.Z_sum(1,2,3) == 6


def test_register_analysis_not_callable():
    """ Exception handle when register_analysis tries to store a non-callable """
    pass


def test_register_plot():

    @Temporary.register_plot()
    def B_plot():
        pass

    assert "B_plot" in Temporary.__dict__
