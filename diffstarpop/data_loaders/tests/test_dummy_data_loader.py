"""
"""

import os

import pytest

try:
    assert os.path.isfile("/path/to/testing_data")
    HAS_SOME_DATA = True
except AssertionError:
    HAS_SOME_DATA = False
NO_DATA_MSG = "Test only runs on machines where data is available"


def test_dummy_data_loader_imports():
    from .. import dummy_data_loader as ddl  # noqa


@pytest.mark.skipif(not HAS_SOME_DATA, reason=NO_DATA_MSG)
def test_dummy_data_loader_works_on_machines_where_data_exists():
    from .. import dummy_data_loader as ddl  # noqa
