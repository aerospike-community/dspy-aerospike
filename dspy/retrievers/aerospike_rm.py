from __future__ import annotations

import heapq
import math
from array import array
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import dspy
from dspy.dsp.utils import dotdict
from dspy.utils.callback import BaseCallback

if TYPE_CHECKING:
    from aerospike_sdk import RecordResult
    from aerospike_sdk.sync import Cluster


class AerospikeRM(dspy.Retrieve):
    """Read-only exact cosine retrieval from a caller-populated Aerospike set.

    Each independent passage record must have a STRING bin (``text_key``,
    default ``content``) and a native FLOAT32 Vector bin (``vector_key``, default
    ``embedding``), not a list or blob. All stored vectors must be finite,
    nonzero, and have the same dimension and embedding model as query vectors.
    Every matching record must satisfy this caller-maintained schema. Malformed
    returned rows raise and backend errors propagate; this is not a corpus
    validator, since Top-K can exclude malformed lower-ranked records.

    Pass an already-connected ``aerospike_sdk.sync.Cluster`` and a batch
    ``embedder(list[str])`` callable, such as ``dspy.Embedder``, returning one
    finite, nonzero FLOAT32-representable vector per query (lists or a 2D NumPy array).
    Namespace, set, bin names, default ``k``, and callbacks are explicit constructor
    arguments. One synchronous session is created and reused; the caller owns
    the cluster and must keep it open for retrieval and close it afterward.
    This module does not ingest data, manage keys/TTL, or create collections or
    secondary indexes; transactions, SC/MRT, and Graph are not required.

    Requires an SDK/server supporting native VECTOR expressions and Top-K.
    Each nonempty query scans the full matching set: O(N*d) cosine work, not
    ANN. For k <= 1000, server Top-K returns the best scores; larger k streams
    O(N) projected text/score payload and uses an O(k) client heap. Score ties
    have unspecified order. A list of queries is embedded once and its top-k
    results are flattened in query order; empty strings are ignored. ``k=0``
    performs no retrieval. ``forward`` optionally accepts a prebuilt
    ``filter_expression``; other keyword arguments are rejected.

    Example with a caller-owned connected ``cluster`` and matching ``embedder``::

        rm = AerospikeRM("test", "passages", cluster, embedder, k=3)
        with dspy.context(rm=rm):
            passages = dspy.Retrieve(k=2)("What is DSPy?").passages

    Direct calls return ``dotdict(long_text=text, score=cosine_similarity)`` rows.
    """

    def __init__(
        self,
        namespace: str,
        set_name: str,
        aerospike_client: Cluster,
        embedder: Callable[[list[str]], Iterable[Iterable[float]]],
        k: int = 3,
        text_key: str = "content",
        vector_key: str = "embedding",
        *,
        callbacks: list[BaseCallback] | None = None,
    ):
        for name, value in (("namespace", namespace), ("set_name", set_name)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")

        for name, value in (("text_key", text_key), ("vector_key", vector_key)):
            message = f"{name} must be a UTF-8 bin name of 1 to 15 bytes"
            if not isinstance(value, str):
                raise ValueError(message)
            try:
                size = len(value.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(message) from error
            if not 1 <= size <= 15:
                raise ValueError(message)

        if text_key == vector_key:
            raise ValueError("text_key and vector_key must be distinct bin names")
        if not callable(embedder):
            raise TypeError("embedder must be callable")
        self._validate_k(k)

        try:
            from aerospike_sdk import DataSet
            from aerospike_sdk.sync import Cluster
        except ImportError as error:
            raise ImportError("AerospikeRM requires the 'aerospike' extra (aerospike-sdk) on Python 3.11+.") from error

        if not isinstance(aerospike_client, Cluster):
            raise TypeError("aerospike_client must be a synchronous aerospike_sdk.sync.Cluster")

        super().__init__(k=k, callbacks=callbacks)
        self._namespace = namespace
        self._set_name = set_name
        self._aerospike_client = aerospike_client
        self._embedder = embedder
        self._text_key = text_key
        self._vector_key = vector_key
        self._score_name = "_dspy_score"
        while self._score_name in (text_key, vector_key):
            self._score_name += "_"
        self._dataset = DataSet.of(namespace, set_name)
        self._session = aerospike_client.create_session()

    @staticmethod
    def _validate_k(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be a nonnegative integer")
        if k < 0:
            raise ValueError("k must be a nonnegative integer")

    def _embed_queries(self, queries: list[str]) -> list[list[float]]:
        output = self._embedder(queries)
        if isinstance(output, (str, bytes, bytearray, Mapping)):
            raise ValueError("embedder must return one embedding vector per query")
        try:
            embeddings = list(output)
        except TypeError as error:
            raise ValueError("embedder must return one embedding vector per query") from error
        if len(embeddings) != len(queries):
            raise ValueError("embedder must return one embedding vector per query")

        vectors = []
        dimension = None
        for embedding in embeddings:
            message = "Each embedding must be a nonempty vector of finite numeric values"
            if isinstance(embedding, (str, bytes, bytearray, Mapping)):
                raise ValueError(message)
            try:
                values = list(embedding)
            except TypeError as error:
                raise ValueError(message) from error
            if not values or any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise ValueError(message)
            try:
                vector = list(array("f", (float(value) for value in values)))
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(message) from error
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(message)
            if not any(vector):
                raise ValueError("Each embedding must be a nonzero vector")
            if dimension is not None and len(vector) != dimension:
                raise ValueError("All embeddings must have the same dimension")
            dimension = len(vector)
            vectors.append(vector)
        return vectors

    def _decode_row(self, row: RecordResult) -> dotdict:
        bins = row.record_or_raise().bins
        text = bins.get(self._text_key)
        if not isinstance(text, str):
            raise ValueError(f"Stored bin {self._text_key!r} must contain a string")
        score = bins.get(self._score_name)
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
            raise ValueError("Projected cosine score must be a finite number")
        return dotdict({"long_text": text, "score": score})

    def forward(self, query: str | list[str], k: int | None = None, **kwargs) -> list[Any]:
        effective_k = self.k if k is None else k
        self._validate_k(effective_k)
        if effective_k == 0:
            return []

        filter_expression = kwargs.pop("filter_expression", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(kwargs))}")
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list) and all(isinstance(item, str) for item in query):
            queries = query
        else:
            raise TypeError("query must be a string or a list of strings")
        queries = [item for item in queries if item]
        if not queries:
            return []
        embeddings = self._embed_queries(queries)

        from aerospike_async import ExpOperation, ExpReadFlags, Operation
        from aerospike_sdk import Exp, Order, OrderByType, Vector

        passages = []
        for embedding in embeddings:
            builder = self._session.query(self._dataset).with_op_projection(
                Operation.get_bin(self._text_key),
                ExpOperation.read(
                    self._score_name,
                    Exp.cosine_similarity(Vector(embedding), Exp.vector_bin(self._vector_key)),
                    ExpReadFlags.EVAL_NO_FAIL,
                ),
            )
            if filter_expression is not None:
                builder = builder.filter_expression(filter_expression)
            if effective_k <= 1000:
                builder = builder.order_by(self._score_name, OrderByType.DOUBLE, Order.DESC).top_k(effective_k)
            stream = builder.execute()
            try:
                decoded = (self._decode_row(row) for row in stream)
                if effective_k <= 1000:
                    passages.extend(decoded)
                else:
                    passages.extend(heapq.nlargest(effective_k, decoded, key=itemgetter("score")))
            finally:
                stream.close()
        return passages
