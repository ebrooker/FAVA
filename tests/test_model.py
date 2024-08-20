from fava.model import Model
import pytest


def test_model_nonexistent_dir(temp_model):
    temp_dir = (
        "abcdefghijklmnopqrstuvwxyz/this_directory_right_here_does_not_exist_at_all"
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        model = temp_model(directory=temp_dir)
    assert str(exc_info.value) == f"Cannot find model directory: {temp_dir}"


def test_model_empty_dir(temp_model, temp_dir):
    with pytest.raises(FileNotFoundError) as exc_info:
        model = temp_model(directory=temp_dir)
    assert str(exc_info.value) == f"The model directory is empty: {temp_dir}"

def test_model_registered_meshes(temp_model):
    meshes = ("Structured", "Unstructured", "FlashAMR")
    mesh_names = Model.mesh_names()
    for mesh in meshes:
        assert mesh in mesh_names

