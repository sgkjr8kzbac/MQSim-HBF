import os
from pathlib import Path

import pytest

from hbf.simulator import build


def pytest_addoption(parser):
    parser.addoption("--evidence-dir", type=Path, help="New directory to retain traces and results")


@pytest.fixture(scope="session")
def binary():
    return build()


@pytest.fixture(scope="session")
def evidence(pytestconfig, tmp_path_factory):
    path = pytestconfig.getoption("--evidence-dir")
    if path is None:
        return tmp_path_factory.mktemp("evidence")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()
