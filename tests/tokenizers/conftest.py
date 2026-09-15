from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tokin.templates import TEMPLATES

HERE = Path(__file__).parent
CASES = [(template, model) for template in TEMPLATES.values() for model in template.models]


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(pytest.mark.tokenizer)


@pytest.fixture(scope="module", params=CASES, ids=[model for _, model in CASES])
def template(request):
    cls, model = request.param
    return cls(AutoTokenizer.from_pretrained(model))
