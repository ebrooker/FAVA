#!/usr/bin/env python

"""Tests for `fava` package."""

# import pytest

# import fava.mesh
# import fava.io as io

# @fava.register_mesh("ChildMesh")
# class ChildMesh:
#     def __init__(self, val):
#         self.val = val
#         super().__init__()

# def test_register_mesh():
#     assert "ChildMesh" in io.__meshes

# def test_build():
#     pi = 3.1415
#     m = fava.load("ChildMesh", val=pi)
#     assert m.val == pi