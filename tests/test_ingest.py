import pytest

from ingest import chunk_page


class WordTokenizer:
    def __call__(self, text, **kwargs):
        offsets, position = [], 0
        for word in text.split(" "):
            offsets.append((position, position + len(word)))
            position += len(word) + 1
        return {"offset_mapping": offsets}


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_page_is_one_chunk():
    chunks = chunk_page(words(5), WordTokenizer(), size=10, overlap=2)
    assert chunks == [(words(5), 5)]


def test_chunks_respect_size_and_overlap():
    chunks = chunk_page(words(25), WordTokenizer(), size=10, overlap=2)
    texts = [text.split() for text, _ in chunks]
    assert [len(t) for t in texts] == [10, 10, 9]
    assert texts[0][-2:] == texts[1][:2]
    assert texts[1][-2:] == texts[2][:2]


def test_token_counts_match_chunk_lengths():
    for text, tokens in chunk_page(words(25), WordTokenizer(), size=10, overlap=2):
        assert len(text.split()) == tokens


def test_last_chunk_ends_exactly_at_the_page_end():
    text = words(23)
    chunks = chunk_page(text, WordTokenizer(), size=10, overlap=3)
    assert text.endswith(chunks[-1][0])


def test_every_word_is_covered():
    text = words(47)
    covered = {w for chunk, _ in chunk_page(text, WordTokenizer(), size=12, overlap=4) for w in chunk.split()}
    assert covered == set(text.split())


@pytest.mark.parametrize("size,overlap", [(300, 60), (500, 100), (800, 160)])
def test_benchmark_settings_produce_valid_chunks(size, overlap):
    chunks = chunk_page(words(2000), WordTokenizer(), size=size, overlap=overlap)
    assert all(tokens <= size for _, tokens in chunks)
    assert chunks[-1][0].endswith("w1999")
