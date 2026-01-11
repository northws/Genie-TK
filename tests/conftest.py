'''
Pytest configuration for Genie-TK tests

License: MIT
'''

import pytest
import torch


def pytest_configure(config):
    #Configure pytest.#
    config.addinivalue_line(
        "markers", "cuda: marks tests as requiring CUDA"
    )


def pytest_collection_modifyitems(config, items):
    #Skip CUDA tests if CUDA is not available.#
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
