import pytest

import pathlib
import tempfile

from typing import Optional

from fava.model import Model

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

# Clean up: Remove the temporary files after the tests  
@pytest.fixture(autouse=True)  
def cleanup_temp_files(temp_dir):  
    yield  
    for file in temp_dir.iterdir():  
        if file.is_file():  
            file.unlink()


@pytest.fixture
def temp_model():
    def _temp_model(directory: str, name: Optional[str]=None, frontend: Optional[str]=None):
        return Model(directory=directory, name=name, frontend=frontend)
    yield _temp_model