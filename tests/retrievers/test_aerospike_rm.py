import heapq
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, create_autospec, patch

import pytest

import dspy
from dspy.dsp.utils import dotdict
from dspy.retrievers.aerospike_rm import AerospikeRM
from dspy.utils.callback import BaseCallback


@pytest.fixture
def client():
    sdk = pytest.importorskip("aerospike_sdk.sync")
    cluster = create_autospec(sdk.Cluster, instance=True, spec_set=True)
    cluster.create_session.return_value = create_autospec(sdk.Session, instance=True, spec_set=True, name="session")
    return cluster


@pytest.fixture
def query_api(client, monkeypatch):
    sdk = pytest.importorskip("aerospike_sdk")
    native = pytest.importorskip("aerospike_async")
    sync = pytest.importorskip("aerospike_sdk.sync")
    stream_module = pytest.importorskip("aerospike_sdk.sync.record_stream")
    builder = create_autospec(sync.QueryBuilder, instance=True, spec_set=True, name="query")
    for method in ("with_op_projection", "filter_expression", "order_by", "top_k"):
        getattr(builder, method).return_value = builder
    stream = MagicMock(spec_set=stream_module.RecordStream, name="stream")
    stream.__iter__.return_value = iter([])
    builder.execute.return_value = stream
    client.create_session.return_value.query.return_value = builder
    operation = Mock(spec_set=["get_bin"])
    exp_operation = Mock(spec_set=["read"])
    expression = Mock(spec_set=["cosine_similarity", "vector_bin"])
    vector = Mock()
    monkeypatch.setattr(native, "Operation", operation)
    monkeypatch.setattr(native, "ExpOperation", exp_operation)
    monkeypatch.setattr(sdk, "Exp", expression)
    monkeypatch.setattr(sdk, "Vector", vector)
    return SimpleNamespace(
        builder=builder,
        stream=stream,
        operation=operation,
        exp_operation=exp_operation,
        expression=expression,
        vector=vector,
        order=sdk.Order,
        order_type=sdk.OrderByType,
    )


@pytest.fixture
def embedder():
    return Mock(return_value=[[1.0, 0.0]])


def make_row(retriever, text="passage", score=0.9):
    row = Mock(spec_set=["record_or_raise"])
    row.record_or_raise.return_value = SimpleNamespace(bins={retriever._text_key: text, retriever._score_name: score})
    return row


def test_constructor_initializes_base_and_creates_one_session(client, embedder):
    retriever = AerospikeRM("test", "documents", client, embedder)

    assert isinstance(retriever, dspy.Retrieve)
    assert retriever.k == 3
    assert retriever.callbacks == []
    assert isinstance(retriever.stage, str)
    assert len(retriever.stage) == 16
    assert retriever._namespace == "test"
    assert retriever._set_name == "documents"
    assert retriever._aerospike_client is client
    assert retriever._embedder is embedder
    assert retriever._text_key == "content"
    assert retriever._vector_key == "embedding"
    assert retriever._session is client.create_session.return_value
    assert retriever._dataset.namespace == "test"
    assert retriever._dataset.set_name == "documents"
    client.create_session.assert_called_once_with()
    assert client.mock_calls == [call.create_session()]
    embedder.assert_not_called()


@pytest.mark.parametrize("field", ["namespace", "set_name"])
@pytest.mark.parametrize("value", ["", "   ", None, 1, b"test"])
def test_rejects_invalid_namespace_and_set_before_creating_session(client, embedder, field, value):
    arguments = {"namespace": "test", "set_name": "documents"}
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        AerospikeRM(**arguments, aerospike_client=client, embedder=embedder)

    client.create_session.assert_not_called()
    embedder.assert_not_called()


@pytest.mark.parametrize("field", ["text_key", "vector_key"])
@pytest.mark.parametrize("value", ["", "a" * 16, "é" * 8, "\ud800", None, 1, b"content"])
def test_rejects_invalid_utf8_bin_names(client, embedder, field, value):
    with pytest.raises(ValueError, match=rf"{field}.*UTF-8.*1.*15"):
        AerospikeRM("test", "documents", client, embedder, **{field: value})

    client.create_session.assert_not_called()


@pytest.mark.parametrize(
    ("text_key", "vector_key"),
    [("t", "v"), ("t" * 15, "v" * 15), ("é" * 7 + "t", "😀" * 3 + "vec")],
)
def test_accepts_bin_names_at_utf8_byte_boundaries(client, embedder, text_key, vector_key):
    retriever = AerospikeRM("test", "documents", client, embedder, text_key=text_key, vector_key=vector_key)

    assert retriever._text_key == text_key
    assert retriever._vector_key == vector_key
    client.create_session.assert_called_once_with()


def test_rejects_identical_bin_names(client, embedder):
    with pytest.raises(ValueError, match="distinct"):
        AerospikeRM("test", "documents", client, embedder, text_key="same", vector_key="same")

    client.create_session.assert_not_called()


@pytest.mark.parametrize("embedder", [None, "embedder", [], 3])
def test_requires_callable_embedder(client, embedder):
    with pytest.raises(TypeError, match=r"embedder.*callable"):
        AerospikeRM("test", "documents", client, embedder)

    client.create_session.assert_not_called()


@pytest.mark.parametrize(
    ("k", "error"),
    [(-1, ValueError), (True, TypeError), (False, TypeError), (1.5, TypeError), ("3", TypeError), (None, TypeError)],
)
def test_rejects_invalid_default_k(client, embedder, k, error):
    with pytest.raises(error, match=r"k.*nonnegative integer"):
        AerospikeRM("test", "documents", client, embedder, k=k)

    client.create_session.assert_not_called()


@pytest.mark.parametrize("k", [0, 1, 10])
def test_accepts_nonnegative_integer_k(client, embedder, k):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)

    assert retriever.k == k
    client.create_session.assert_called_once_with()


def test_requires_synchronous_sdk_cluster(client, embedder):
    with pytest.raises(TypeError, match=r"aerospike_client.*synchronous.*Cluster"):
        AerospikeRM("test", "documents", object(), embedder)


def test_session_creation_errors_propagate_without_closing_caller_client(client, embedder):
    error = RuntimeError("session unavailable")
    client.create_session.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        AerospikeRM("test", "documents", client, embedder)

    assert raised.value is error
    client.create_session.assert_called_once_with()
    client.close.assert_not_called()


def test_constructor_does_not_read_environment(client, embedder):
    with (
        patch("os.getenv", side_effect=AssertionError("Unexpected environment access")),
        patch.object(type(os.environ), "__getitem__", side_effect=AssertionError("Unexpected environment access")),
    ):
        retriever = AerospikeRM("test", "documents", client, embedder)

    assert retriever._session is client.create_session.return_value
    client.create_session.assert_called_once_with()


def test_zero_k_returns_empty_without_new_sessions_or_embedding(client, embedder):
    retriever = AerospikeRM("test", "documents", client, embedder, k=0)
    session = retriever._session

    assert retriever("first") == []
    assert retriever(["second", "third"]) == []
    assert retriever.forward("fourth", k=0, unused=True) == []
    assert retriever._session is session
    client.create_session.assert_called_once_with()
    client.close.assert_not_called()
    embedder.assert_not_called()


def test_per_call_zero_k_does_not_change_default(client, embedder):
    retriever = AerospikeRM("test", "documents", client, embedder, k=2)

    assert retriever("query", k=0) == []
    assert retriever.k == 2
    client.create_session.assert_called_once_with()
    embedder.assert_not_called()


@pytest.mark.parametrize("k", [-1, True, False, 1.5, "3"])
def test_rejects_invalid_per_call_k_without_new_session(client, embedder, k):
    retriever = AerospikeRM("test", "documents", client, embedder)

    with pytest.raises((TypeError, ValueError), match=r"k.*nonnegative integer"):
        retriever("query", k=k)

    client.create_session.assert_called_once_with()
    client.close.assert_not_called()
    embedder.assert_not_called()


def test_inherited_call_preserves_callback_lifecycle(client, embedder):
    callback = Mock(spec=BaseCallback)
    retriever = AerospikeRM("test", "documents", client, embedder, callbacks=[callback])

    assert AerospikeRM.__call__ is dspy.Retrieve.__call__
    assert retriever("query", k=0) == []
    callback.on_module_start.assert_called_once()
    callback.on_module_end.assert_called_once()
    start = callback.on_module_start.call_args.kwargs
    end = callback.on_module_end.call_args.kwargs
    assert start["instance"] is retriever
    assert start["call_id"] == end["call_id"]
    assert end["outputs"] == []
    assert end["exception"] is None


def test_inherited_call_reports_forward_errors_to_callbacks(client, embedder):
    callback = Mock(spec=BaseCallback)
    retriever = AerospikeRM("test", "documents", client, embedder, callbacks=[callback])
    error = RuntimeError("embedding service unavailable")
    embedder.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        retriever("query")

    assert raised.value is error
    retriever._session.query.assert_not_called()
    callback.on_module_start.assert_called_once()
    callback.on_module_end.assert_called_once()
    end = callback.on_module_end.call_args.kwargs
    assert end["outputs"] is None
    assert end["exception"] is raised.value


@pytest.mark.parametrize("k", [1, 3, 1000])
def test_projects_text_and_server_cosine_topk(client, embedder, query_api, k):
    from aerospike_async import ExpReadFlags

    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    expected = [dotdict(long_text="best", score=0.9), dotdict(long_text="next", score=0.7)][:k]
    rows = [make_row(retriever, item.long_text, item.score) for item in expected]
    query_api.stream.__iter__.return_value = iter(rows)

    results = retriever("question")

    assert results == expected
    assert all(isinstance(result, dotdict) for result in results)
    assert all(isinstance(result.long_text, str) for result in results)
    for row in rows:
        row.record_or_raise.assert_called_once_with()
    embedder.assert_called_once_with(["question"])
    retriever._session.query.assert_called_once_with(retriever._dataset)
    query_api.vector.assert_called_once_with([1.0, 0.0])
    query_api.expression.vector_bin.assert_called_once_with("embedding")
    query_api.expression.cosine_similarity.assert_called_once_with(
        query_api.vector.return_value, query_api.expression.vector_bin.return_value
    )
    query_api.operation.get_bin.assert_called_once_with("content")
    query_api.exp_operation.read.assert_called_once_with(
        retriever._score_name, query_api.expression.cosine_similarity.return_value, ExpReadFlags.EVAL_NO_FAIL
    )
    assert query_api.builder.mock_calls == [
        call.with_op_projection(query_api.operation.get_bin.return_value, query_api.exp_operation.read.return_value),
        call.order_by(retriever._score_name, query_api.order_type.DOUBLE, query_api.order.DESC),
        call.top_k(k),
        call.execute(),
    ]
    query_api.stream.close.assert_called_once_with()
    client.create_session.assert_called_once_with()
    client.close.assert_not_called()


def test_works_as_unchanged_dspy_retrieve_backend(client, embedder, query_api):
    retriever = AerospikeRM("test", "documents", client, embedder)
    query_api.stream.__iter__.return_value = iter([make_row(retriever, "retrieved passage")])

    with dspy.context(rm=retriever):
        prediction = dspy.Retrieve(k=1)("question")

    assert prediction.passages == ["retrieved passage"]
    query_api.builder.top_k.assert_called_once_with(1)
    assert retriever.k == 3


@pytest.mark.parametrize(
    ("text_key", "vector_key"),
    [("body", "vector"), ("_dspy_score", "embedding"), ("content", "_dspy_score"), ("_dspy_score", "_dspy_score_")],
)
def test_configured_bins_do_not_collide_with_projected_score(client, embedder, query_api, text_key, vector_key):
    retriever = AerospikeRM("test", "documents", client, embedder, text_key=text_key, vector_key=vector_key)
    query_api.stream.__iter__.return_value = iter([make_row(retriever, "custom text")])

    assert retriever("question")[0].long_text == "custom text"

    assert retriever._score_name not in (text_key, vector_key)
    assert 1 <= len(retriever._score_name.encode("utf-8")) <= 15
    query_api.operation.get_bin.assert_called_once_with(text_key)
    query_api.expression.vector_bin.assert_called_once_with(vector_key)


def test_batches_embedding_and_flattens_results_in_query_order(client, embedder, query_api):
    embedder.return_value = [[1, 0], [0, 1]]
    retriever = AerospikeRM("test", "documents", client, embedder, k=2)
    first_rows = [make_row(retriever, "first best", 1.0), make_row(retriever, "first second", 0.6)]
    second_rows = [make_row(retriever, "second best", 0.8), make_row(retriever, "second second", 0.4)]
    query_api.stream.__iter__.side_effect = [iter(first_rows), iter(second_rows)]

    results = retriever(["", "first query", "", "second query", ""])

    assert [result.long_text for result in results] == ["first best", "first second", "second best", "second second"]
    embedder.assert_called_once_with(["first query", "second query"])
    assert query_api.vector.call_args_list == [call([1.0, 0.0]), call([0.0, 1.0])]
    assert retriever._session.query.call_count == 2
    assert query_api.stream.close.call_count == 2
    client.create_session.assert_called_once_with()
    client.close.assert_not_called()


@pytest.mark.parametrize("queries", ["", [], ["", ""]])
def test_empty_queries_do_not_embed_or_query(client, embedder, query_api, queries):
    retriever = AerospikeRM("test", "documents", client, embedder)

    assert retriever(queries) == []

    embedder.assert_not_called()
    retriever._session.query.assert_not_called()


def test_whitespace_query_is_not_discarded(client, embedder, query_api):
    retriever = AerospikeRM("test", "documents", client, embedder)

    assert retriever(" ") == []

    embedder.assert_called_once_with([" "])


def test_per_call_k_override_and_repeated_calls_reuse_session(client, embedder, query_api):
    retriever = AerospikeRM("test", "documents", client, embedder, k=2)
    query_api.stream.__iter__.side_effect = [
        iter([make_row(retriever)]),
        iter([make_row(retriever), make_row(retriever)]),
    ]
    session = retriever._session

    assert len(retriever("first", k=1)) == 1
    assert len(retriever("second")) == 2

    assert query_api.builder.top_k.call_args_list == [call(1), call(2)]
    assert retriever.k == 2
    assert retriever._session is session
    client.create_session.assert_called_once_with()
    client.close.assert_not_called()
    assert session.mock_calls == [call.query(retriever._dataset), call.query(retriever._dataset)]
    assert query_api.stream.close.call_count == 2


@pytest.mark.parametrize("k", [1001, 1200, 2500])
def test_large_k_streams_all_server_scores_into_bounded_heap_without_truncation(client, embedder, query_api, k):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    rows = [make_row(retriever, str(index), index / 2000) for index in range(1600)]
    query_api.stream.__iter__.return_value = iter(rows)

    with patch("heapq.nlargest", wraps=heapq.nlargest) as largest:
        results = retriever("question")

    assert len(results) == min(k, 1600)
    assert [result.long_text for result in results] == [str(index) for index in range(1599, max(-1, 1599 - k), -1)]
    query_api.builder.order_by.assert_not_called()
    query_api.builder.top_k.assert_not_called()
    query_api.builder.with_op_projection.assert_called_once()
    query_api.builder.execute.assert_called_once_with()
    largest.assert_called_once()
    assert largest.call_args.args[0] == k
    assert not isinstance(largest.call_args.args[1], (list, tuple))
    for row in rows:
        row.record_or_raise.assert_called_once_with()
    query_api.stream.close.assert_called_once_with()


@pytest.mark.parametrize("k", [3, 1001])
def test_optional_filter_expression_is_routed_before_execution(client, embedder, query_api, k):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    predicate = object()

    assert retriever("question", filter_expression=predicate) == []

    query_api.builder.filter_expression.assert_called_once_with(predicate)
    calls = query_api.builder.mock_calls
    assert calls.index(call.filter_expression(predicate)) < calls.index(call.execute())


def test_none_filter_is_not_applied(client, embedder, query_api):
    retriever = AerospikeRM("test", "documents", client, embedder)

    assert retriever("question", filter_expression=None) == []

    query_api.builder.filter_expression.assert_not_called()


@pytest.mark.parametrize("query", ["question", "", []])
def test_unknown_kwargs_are_rejected_before_embedding_or_querying(client, embedder, query):
    retriever = AerospikeRM("test", "documents", client, embedder)

    with pytest.raises(TypeError, match="unused"):
        retriever(query, unused=True)

    retriever._session.query.assert_not_called()
    embedder.assert_not_called()


@pytest.mark.parametrize(
    "query", [None, 1, b"query", ("query",), {"query"}, {"q": "query"}, [None], ["", 0], ["ok", False]]
)
def test_rejects_invalid_query_types_before_embedding(client, embedder, query):
    retriever = AerospikeRM("test", "documents", client, embedder)

    with pytest.raises(TypeError, match=r"query|strings"):
        retriever(query)

    retriever._session.query.assert_not_called()
    embedder.assert_not_called()


@pytest.mark.parametrize(
    ("queries", "embeddings"),
    [
        (["q"], None),
        (["q"], 1),
        (["q"], []),
        (["q"], "embedding"),
        (["q"], b"embedding"),
        (["q"], {"vector": [1, 0]}),
        (["q"], [1.0, 0.0]),
        (["q"], [[1.0, 0.0], [1.0, 0.0]]),
        (["q"], [None]),
        (["q"], [[]]),
        (["q"], [b"\x01\x02"]),
        (["q"], [bytearray([1, 2])]),
        (["q"], [{1: 1.0, 2: 0.0}]),
        (["q"], [[0, 0]]),
        (["q"], [["1", 0]]),
        (["q"], [[True, 0]]),
        (["q"], [[None, 0]]),
        (["q"], [[[1], 0]]),
        (["q"], [[1 + 2j, 0]]),
        (["q"], [[float("nan"), 0]]),
        (["q"], [[float("inf"), 0]]),
        (["q"], [[float("-inf"), 0]]),
        (["q"], [[1e40, 1.0]]),
        (["q"], [[1e-50, 0.0]]),
        (["first", "second"], [[1, 0]]),
        (["first", "second"], [[1, 0], [1, 0, 0]]),
        (["first", "second"], [[1, 0], [0, 0]]),
        (["first", "second"], [[1, 0], [float("nan"), 0]]),
    ],
)
def test_bad_embedding_batches_fail_before_any_query(client, embedder, queries, embeddings):
    embedder.return_value = embeddings
    retriever = AerospikeRM("test", "documents", client, embedder)

    with pytest.raises(ValueError, match="embed"):
        retriever(queries)

    embedder.assert_called_once_with(queries)
    retriever._session.query.assert_not_called()
    client.close.assert_not_called()


def test_supports_dspy_embedder_numpy_batch_output(client, query_api):
    np = pytest.importorskip("numpy")
    model = Mock(return_value=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    embedder = dspy.Embedder(model)
    retriever = AerospikeRM("test", "documents", client, embedder)
    query_api.stream.__iter__.side_effect = [iter([]), iter([])]

    assert retriever(["first", "", "second"]) == []

    model.assert_called_once_with(["first", "second"])
    assert query_api.vector.call_args_list == [call([1.0, 0.0]), call([0.0, 1.0])]


@pytest.mark.parametrize("k", [3, 1001])
@pytest.mark.parametrize("text", [None, 42, b"passage", ["passage"]])
def test_malformed_stored_text_raises_and_closes_stream(client, embedder, query_api, k, text):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    query_api.stream.__iter__.return_value = iter([make_row(retriever, text=text)])

    with pytest.raises(ValueError, match=r"content.*string"):
        retriever("question")

    query_api.stream.close.assert_called_once_with()
    client.close.assert_not_called()


@pytest.mark.parametrize("k", [3, 1001])
@pytest.mark.parametrize("score", [None, "0.9", True, float("nan"), float("inf"), float("-inf")])
def test_malformed_stored_score_raises_and_closes_stream(client, embedder, query_api, k, score):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    query_api.stream.__iter__.return_value = iter([make_row(retriever, score=score)])

    with pytest.raises(ValueError, match=r"score.*finite"):
        retriever("question")

    query_api.stream.close.assert_called_once_with()


@pytest.mark.parametrize("missing", ["text", "score"])
def test_missing_projected_bins_raise_instead_of_skipping_rows(client, embedder, query_api, missing):
    retriever = AerospikeRM("test", "documents", client, embedder)
    row = make_row(retriever)
    key = retriever._text_key if missing == "text" else retriever._score_name
    del row.record_or_raise.return_value.bins[key]
    query_api.stream.__iter__.return_value = iter([row])

    with pytest.raises(ValueError):
        retriever("question")

    query_api.stream.close.assert_called_once_with()


@pytest.mark.parametrize("k", [3, 1001])
def test_per_row_backend_errors_propagate_and_close_stream(client, embedder, query_api, k):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    error = RuntimeError("vector expression failed")
    row = Mock(spec_set=["record_or_raise"])
    row.record_or_raise.side_effect = error
    query_api.stream.__iter__.return_value = iter([make_row(retriever), row])

    with pytest.raises(RuntimeError) as raised:
        retriever("question")

    assert raised.value is error
    row.record_or_raise.assert_called_once_with()
    query_api.stream.close.assert_called_once_with()
    client.close.assert_not_called()


@pytest.mark.parametrize("k", [3, 1001])
def test_iteration_errors_propagate_and_close_stream(client, embedder, query_api, k):
    retriever = AerospikeRM("test", "documents", client, embedder, k=k)
    error = RuntimeError("scan interrupted")

    def interrupted():
        yield make_row(retriever)
        raise error

    query_api.stream.__iter__.return_value = interrupted()

    with pytest.raises(RuntimeError) as raised:
        retriever("question")

    assert raised.value is error
    query_api.stream.close.assert_called_once_with()
    client.close.assert_not_called()


def test_execute_errors_propagate_without_closing_caller_cluster(client, embedder, query_api):
    retriever = AerospikeRM("test", "documents", client, embedder)
    error = RuntimeError("query unavailable")
    query_api.builder.execute.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        retriever("question")

    assert raised.value is error
    query_api.stream.close.assert_not_called()
    client.close.assert_not_called()


def test_empty_corpus_returns_empty_and_closes_stream(client, embedder, query_api):
    retriever = AerospikeRM("test", "documents", client, embedder)

    assert retriever("question") == []

    query_api.stream.close.assert_called_once_with()


def test_zero_k_ignores_query_validation_and_kwargs_without_io(client, embedder):
    retriever = AerospikeRM("test", "documents", client, embedder)

    assert retriever(None, k=0, unused=True) == []

    retriever._session.query.assert_not_called()
    embedder.assert_not_called()


def test_core_and_retriever_import_without_optional_sdk():
    source = """
import importlib.abc
import sys

class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("aerospike_sdk", "aerospike_async") or fullname.startswith(("aerospike_sdk.", "aerospike_async.")):
            raise ModuleNotFoundError("Optional SDK unavailable", name=fullname)

sys.meta_path.insert(0, BlockSDK())
import dspy
from dspy.retrievers.aerospike_rm import AerospikeRM
assert "aerospike_sdk" not in sys.modules
assert "aerospike_async" not in sys.modules
assert not hasattr(dspy, "AerospikeRM")
try:
    AerospikeRM("test", "documents", object(), lambda queries: [[1.0]])
except ImportError as error:
    assert "aerospike" in str(error)
    assert "aerospike-sdk" in str(error)
    assert "3.11" in str(error)
else:
    raise AssertionError("Missing SDK should raise an informative ImportError")
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
