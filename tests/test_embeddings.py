from pathlib import Path

from ragferee.embeddings import EMBEDDING_DIMENSION, Embedder, FastEmbedder


def test_fake_embedder_follows_protocol(fake_embedder: Embedder) -> None:
    vectors = fake_embedder.embed_documents(["first", "second"])

    assert len(vectors) == 2
    assert all(len(vector) == fake_embedder.dimension for vector in vectors)
    assert fake_embedder.embed_query("first") == vectors[0]


def test_fast_embedder_loads_model_lazily(tmp_path: Path) -> None:
    embedder: Embedder = FastEmbedder(tmp_path)

    assert embedder.dimension == EMBEDDING_DIMENSION
    assert list(tmp_path.iterdir()) == []
