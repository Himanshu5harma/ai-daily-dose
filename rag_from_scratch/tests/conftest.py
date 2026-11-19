import pytest

from rag_from_scratch.corpus import DATA_DIR, load_corpus


@pytest.fixture(scope="session")
def real_corpus():
    if not (DATA_DIR / "corpus").is_dir():
        pytest.skip(
            "corpus not copied; run python -m rag_from_scratch.copy_corpus --doclens ../doclens"
        )
    return load_corpus()
