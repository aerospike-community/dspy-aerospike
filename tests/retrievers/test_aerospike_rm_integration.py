import math
import os
import socket
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest

import dspy
from dspy.retrievers.aerospike_rm import AerospikeRM

pytestmark = pytest.mark.skipif(
    os.getenv("DSPY_AEROSPIKE_INTEGRATION") != "1",
    reason="set DSPY_AEROSPIKE_INTEGRATION=1 to run against a native VECTOR/Top-K server",
)

PASSAGES = [
    ("east passage", [1.0, 0.0]),
    ("near passage", [4.0, 3.0]),
    ("north passage", [0.0, 2.0]),
    ("west passage", [-2.0, 0.0]),
]
QUERY_VECTORS = {"east": [1.0, 0.0], "north": [0.0, 1.0], "west": [-1.0, 0.0]}


def records_or_raise(stream):
    try:
        return [row.record_or_raise() for row in stream]
    finally:
        stream.close()


def cosine(left, right):
    return sum(x * y for x, y in zip(left, right, strict=True)) / (
        math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    )


class FixedEmbedder:
    def __init__(self):
        self.calls = []

    def __call__(self, queries):
        self.calls.append(list(queries))
        return [QUERY_VECTORS[query] for query in queries]


class LiveCorpus:
    def __init__(self, sdk, cluster, session):
        self.sdk = sdk
        self.cluster = cluster
        self.session = session
        self.dataset = sdk.DataSet.of("test", f"dspy_it_{uuid4().hex}")
        self.created_keys = []
        self.next_id = 0

    def insert(self, bins):
        keys = self.dataset.ids([f"passage-{self.next_id + index:05d}" for index in range(len(bins))])
        self.next_id += len(bins)
        for start in range(0, len(keys), 256):
            batch_keys = keys[start : start + 256]
            batch_bins = bins[start : start + 256]
            builder = self.session.insert(batch_keys[:1]).put(batch_bins[0])
            for key, values in zip(batch_keys[1:], batch_bins[1:], strict=True):
                builder = builder.insert(key).put(values)
            stream = builder.execute(on_error=self.sdk.ErrorStrategy.IN_STREAM)
            try:
                rows = list(stream)
            finally:
                stream.close()
            self.created_keys.extend(row.key for row in rows if row.is_ok)
            for row in rows:
                row.record_or_raise()
            assert len(rows) == len(batch_keys)
        return keys

    def remove(self, keys):
        if not keys:
            return
        stream = self.session.delete(keys).execute(on_error=self.sdk.ErrorStrategy.IN_STREAM)
        try:
            rows = list(stream)
        finally:
            stream.close()
        for row in rows:
            row.or_raise()
        assert len(rows) == len(keys)
        self.created_keys = [key for key in self.created_keys if key not in keys]
        assert records_or_raise(self.session.query(keys).execute()) == []

    def retriever(self, **kwargs):
        return AerospikeRM(
            "test",
            self.dataset.set_name,
            self.cluster,
            kwargs.pop("embedder", FixedEmbedder()),
            **kwargs,
        )


@pytest.fixture(scope="module")
def sdk():
    import aerospike_sdk

    return aerospike_sdk


@pytest.fixture(scope="module")
def cluster(sdk):
    from aerospike_sdk.sync import ClusterDefinition

    host = os.getenv("AEROSPIKE_HOST", "127.0.0.1")
    port = int(os.getenv("AEROSPIKE_PORT", "23000"))
    deadline = time.monotonic() + 30
    last_error = None
    connected = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
            connected = ClusterDefinition(host, port).using_services_alternate(False).connect()
            probe = connected.create_session()
            assert probe.info("status")["status"] == "ok"
            assert "test" in probe.info().namespaces()
            break
        except Exception as error:
            last_error = error
            if connected is not None:
                connected.close()
                connected = None
            time.sleep(0.1)
    else:
        pytest.fail(f"explicit Aerospike integration target {host}:{port} was not ready: {last_error!r}")
    try:
        version = connected.server_version()
        assert version is not None
        assert (version.major, version.minor, version.patch) >= (8, 1, 3)
        assert connected.supports_query_operations()
        yield connected
    finally:
        connected.close()


@pytest.fixture(scope="module")
def session(cluster):
    return cluster.create_session()


@pytest.fixture
def corpus(sdk, cluster, session):
    live = LiveCorpus(sdk, cluster, session)
    assert len(live.dataset.set_name.encode("utf-8")) <= 63
    try:
        yield live
    finally:
        live.remove(list(live.created_keys))


@pytest.fixture
def passages(corpus):
    corpus.insert([{"content": text, "embedding": corpus.sdk.Vector(vector)} for text, vector in PASSAGES])
    return corpus


def test_server_capabilities_and_native_vector_roundtrip(corpus):
    version = corpus.cluster.server_version()
    builds = corpus.session.info().build()
    assert builds
    assert all(build.startswith(f"{version.major}.{version.minor}.{version.patch}.") for build in builds)
    assert "Enterprise" in corpus.session.info("edition")["edition"]
    features = corpus.session.info("features")["features"].split(";")
    assert "query-order-by" in features
    assert corpus.cluster.supports_query_operations()
    assert corpus.session.info().namespace_details("test").exists

    values = [0.25, -0.5, 1.0]
    keys = corpus.insert([{"content": "native VECTOR", "embedding": corpus.sdk.Vector(values)}])
    records = records_or_raise(corpus.session.query(keys).execute())
    assert len(records) == 1
    vector = records[0].bins["embedding"]
    assert isinstance(vector, corpus.sdk.Vector)
    assert vector.element_type == corpus.sdk.VectorElementType.FLOAT32
    assert vector.dimensions == 3
    assert list(vector.value) == pytest.approx(values, abs=1e-7)
    rm = corpus.retriever(embedder=lambda queries: [values for _ in queries], k=1)
    result = rm("roundtrip")
    assert [row.long_text for row in result] == ["native VECTOR"]
    assert [row.score for row in result] == pytest.approx([1.0], abs=1e-6)


def test_descending_cosine_scores_default_override_and_zero(passages):
    embedder = FixedEmbedder()
    rm = passages.retriever(embedder=embedder)
    session = rm._session

    result = rm("east")
    assert [row.long_text for row in result] == [text for text, _ in PASSAGES[:3]]
    assert [row.score for row in result] == pytest.approx([1.0, 0.8, 0.0], abs=1e-6)
    assert [row.long_text for row in rm("east", k=1)] == ["east passage"]
    assert [row.long_text for row in rm("east", k=4)] == [text for text, _ in PASSAGES]
    assert [row.score for row in rm("east", k=4)] == pytest.approx([1.0, 0.8, 0.0, -1.0], abs=1e-6)
    calls = list(embedder.calls)
    assert rm("east", k=0) == []
    assert passages.retriever(k=0)("east") == []
    assert embedder.calls == calls
    assert rm.k == 3
    assert rm._session is session
    assert len(rm("east")) == 3


def test_dspy_retrieve_context_end_to_end(passages):
    rm = passages.retriever(k=1)
    with dspy.context(rm=rm):
        result = dspy.Retrieve(k=2)("east")

    assert result.passages == ["east passage", "near passage"]
    assert rm.k == 1
    assert passages.session.info("status")["status"] == "ok"


def test_multiquery_flattens_in_query_order_and_embeds_once(passages):
    embedder = FixedEmbedder()
    rm = passages.retriever(embedder=embedder, k=2)

    result = rm(["east", "", "north", "west"])

    assert embedder.calls == [["east", "north", "west"]]
    assert [row.long_text for row in result] == [
        "east passage",
        "near passage",
        "north passage",
        "near passage",
        "west passage",
        "north passage",
    ]
    assert [row.score for row in result] == pytest.approx([1.0, 0.8, 1.0, 0.6, 1.0, 0.0], abs=1e-6)


@pytest.mark.parametrize("query", ["", [], ["", ""]])
def test_empty_queries_do_not_embed(passages, query):
    embedder = FixedEmbedder()
    rm = passages.retriever(embedder=embedder)

    assert rm(query) == []
    assert embedder.calls == []


@pytest.mark.parametrize("k", [1, 3, 1001])
def test_empty_set_returns_no_passages(corpus, k):
    assert corpus.retriever(k=k)("east") == []


@pytest.mark.parametrize("k", [1000, 1001])
def test_k_larger_than_corpus_returns_every_record(passages, k):
    result = passages.retriever(k=k)("east")

    assert [row.long_text for row in result] == [text for text, _ in PASSAGES]
    assert [row.score for row in result] == pytest.approx([1.0, 0.8, 0.0, -1.0], abs=1e-6)


@pytest.mark.parametrize(
    ("text_key", "vector_key"),
    [
        ("texte", "vecteur"),
        ("文", "向量"),
        ("_dspy_score", "vec"),
        ("text", "_dspy_score"),
        ("_dspy_score", "_dspy_score_"),
    ],
)
def test_unicode_custom_bins_and_internal_score_collisions(corpus, text_key, vector_key):
    texts = ["こんにちは 🌍 — café", "Привет κόσμε"]
    bins = [
        {text_key: text, vector_key: corpus.sdk.Vector(vector)}
        for text, vector in zip(texts, [[1.0, 0.0], [3.0, 4.0]], strict=True)
    ]
    for values in bins:
        values.setdefault("_dspy_score", -12345)
    corpus.insert(bins)
    rm = corpus.retriever(text_key=text_key, vector_key=vector_key, k=2)

    result = rm("east")

    assert [row.long_text for row in result] == texts
    assert [row.score for row in result] == pytest.approx([1.0, 0.6], abs=1e-6)
    assert rm._score_name not in (text_key, vector_key)


@pytest.mark.parametrize("k", [3, 1001])
def test_prebuilt_server_filter_applies_before_ranking(corpus, k):
    corpus.insert(
        [
            {"content": "excluded best", "embedding": corpus.sdk.Vector([1.0, 0.0]), "tenant": 2},
            {"content": "allowed near", "embedding": corpus.sdk.Vector([4.0, 3.0]), "tenant": 1},
            {"content": "allowed north", "embedding": corpus.sdk.Vector([0.0, 1.0]), "tenant": 1},
        ]
    )
    expression = corpus.sdk.Exp.eq(corpus.sdk.Exp.int_bin("tenant"), corpus.sdk.Exp.int_val(1))
    rm = corpus.retriever(k=k)

    result = rm("east", filter_expression=expression)

    assert [row.long_text for row in result] == ["allowed near", "allowed north"]
    assert [row.score for row in result] == pytest.approx([0.8, 0.0], abs=1e-6)
    assert len(rm("east")) == 3


@pytest.mark.parametrize("k", [1, 3, 1001])
@pytest.mark.parametrize(
    "malformation",
    [
        "missing_text",
        "numeric_text",
        "missing_vector",
        "list_vector",
        "scalar_vector",
        "wrong_dimension",
        "float64",
        "nan_vector",
        "infinite_vector",
    ],
)
def test_invalid_corpus_raises_and_same_client_session_recovers(corpus, malformation, k):
    sdk = corpus.sdk
    valid = {"content": "valid passage", "embedding": sdk.Vector([1.0, 0.0])}
    if k != 1:
        corpus.insert([valid])
    invalid = {"content": "invalid passage", "embedding": sdk.Vector([0.0, 1.0])}
    if malformation == "missing_text":
        del invalid["content"]
    elif malformation == "numeric_text":
        invalid["content"] = 42
    elif malformation == "missing_vector":
        del invalid["embedding"]
    elif malformation == "list_vector":
        invalid["embedding"] = [0.0, 1.0]
    elif malformation == "scalar_vector":
        invalid["embedding"] = 7
    elif malformation == "wrong_dimension":
        invalid["embedding"] = sdk.Vector([0.0, 1.0, 0.0])
    elif malformation == "float64":
        invalid["embedding"] = sdk.Vector([0.0, 1.0], sdk.VectorElementType.FLOAT64)
    elif malformation == "nan_vector":
        invalid["embedding"] = sdk.Vector([float("nan"), 1.0])
    elif malformation == "infinite_vector":
        invalid["embedding"] = sdk.Vector([float("inf"), 1.0])
    invalid_keys = corpus.insert([invalid])
    rm = corpus.retriever(k=k)
    session = rm._session

    if malformation in ("missing_text", "numeric_text"):
        with pytest.raises(ValueError, match="Stored bin 'content' must contain a string"):
            rm("east")
    else:
        with pytest.raises(ValueError, match="Projected cosine score must be a finite number"):
            rm("east")

    assert corpus.session.info("status")["status"] == "ok"
    assert len(records_or_raise(session.query(corpus.dataset).execute())) == (1 if k == 1 else 2)
    corpus.remove(invalid_keys)
    if k == 1:
        corpus.insert([valid])
    result = rm("east")
    assert [row.long_text for row in result] == ["valid passage"]
    assert [row.score for row in result] == pytest.approx([1.0], abs=1e-6)
    assert rm._session is session
    assert rm._aerospike_client is corpus.cluster
    assert rm.k == k


def test_concurrent_queries_share_cluster_session_and_preserve_k(passages):
    rm = passages.retriever(k=3)
    session = rm._session
    barrier = Barrier(6)
    requests = [("east", 1), ("east", 2), ("east", 4), ("west", 1), ("north", 2), ("east", 1001)] * 4

    def search(request):
        query, k = request
        barrier.wait(timeout=20)
        result = rm(query, k=k)
        expected = sorted(PASSAGES, key=lambda passage: cosine(QUERY_VECTORS[query], passage[1]), reverse=True)[:k]
        assert [row.long_text for row in result] == [text for text, _ in expected]
        assert [row.score for row in result] == pytest.approx(
            [cosine(QUERY_VECTORS[query], vector) for _, vector in expected], abs=1e-6
        )
        assert rm._session is session
        assert rm._aerospike_client is passages.cluster
        return len(result)

    with ThreadPoolExecutor(max_workers=6) as executor:
        counts = list(executor.map(search, requests))

    assert counts == [min(k, len(PASSAGES)) for _, k in requests]
    assert rm.k == 3
    assert rm._session is session
    assert len(rm("east")) == 3


def test_more_than_one_thousand_real_results_are_not_truncated(corpus):
    vectors = []
    for index in range(1105):
        score = 0.95 - index * 1.8 / 1104
        vectors.append(list(array("f", [score, math.sqrt(1.0 - score * score)])))
    texts = [f"passage {index:04d}" for index in range(len(vectors))]
    corpus.insert(
        [{"content": text, "embedding": corpus.sdk.Vector(vector)} for text, vector in zip(texts, vectors, strict=True)]
    )
    expected_scores = [cosine(QUERY_VECTORS["east"], vector) for vector in vectors]
    rm = corpus.retriever(k=1001)

    for k in (1000, 1001, 2000):
        result = rm("east") if k == 1001 else rm("east", k=k)
        count = min(k, len(texts))
        scores = [row.score for row in result]
        assert len(result) == count
        assert len(set(scores)) == count
        assert scores == sorted(scores, reverse=True)
        assert [row.long_text for row in result] == texts[:count]
        assert scores == pytest.approx(expected_scores[:count], abs=2e-6)
    assert rm.k == 1001
