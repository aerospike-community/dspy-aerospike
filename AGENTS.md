# DSPy maintenance notes

General contributor setup and the existing CI commands are documented in
`CONTRIBUTING.md` and `.github/workflows/run_tests.yml`.

## Aerospike retriever contract

`dspy.retrievers.aerospike_rm.AerospikeRM` implements the existing
`dspy.Retrieve` extension point. Import it from its module, like `WeaviateRM`;
do not import optional backend dependencies from DSPy's top-level package.
Existing retrievers, their tests, and the base retrieval interface are
read-only references when maintaining this integration.

The caller supplies an already-connected synchronous SDK `Cluster` and a
batch embedder. The retriever creates one session and reuses it, never connects
per request, never closes the caller's client, and obtains all configuration
from constructor arguments. Use one shared client per process.

### Data contract

| Item | Contract |
| --- | --- |
| Namespace and set | Explicit constructor arguments; the caller populates the set |
| Record granularity | One independently ranked passage per record |
| Keys | Caller-owned; the retriever neither derives nor rewrites them |
| `content` | Default text bin; STRING containing the complete passage |
| `embedding` | Default vector bin; native FLOAT32 VECTOR, not a list or blob |
| Dimensions/model | Identical across stored vectors and the query embedder |
| Bin names | Configurable, distinct, and 1–15 UTF-8 bytes |
| Retention | Caller/server-owned; retrieval never extends or rewrites TTL |
| Indexes | None required or created; there is no ANN/HNSW index |

Separate passage records are necessary because native distance expressions
rank individual VECTOR bins on records. Consolidating the corpus into a CDT
would not preserve that query surface. Ingestion, passage chunking, record-size
limits, and corpus/model migrations remain the existing ingestion system's
responsibility.

Queries compute cosine similarity server-side and project only text and score.
Each query scans the matching set: O(N × dimensions), not sub-linear ANN search.
For `k <= 1000`, Top-K ranking is pushed down. For larger `k`, the same scores
are streamed into a bounded client heap; this preserves results but transfers
O(N) text/score data. Ties have unspecified order.

Direct calls return passage objects with `long_text` and `score`. Through
`dspy.context(rm=rm)`, the unchanged `dspy.Retrieve` returns a `Prediction` with
`passages`. Lists of queries are embedded once and their results flattened in
query order. Empty strings are ignored and `k=0` does no retrieval.

Stored vectors must be finite and nonzero. Returned malformed text/scores fail
explicitly, but retrieval is not whole-corpus validation: Top-K can exclude a
malformed lower-ranked record. Query vectors are validated after FLOAT32
conversion. Backend failures propagate; do not add blind retries or weaken
ordering/count assertions.

The interface is read-only and has no atomic multi-record mutation or snapshot
contract. Transactions, SC, and MRT are unnecessary; the test namespace is AP.
This does not establish transactional guarantees for a future ingestion API.

### Usage

The optional extra is `aerospike`, with the released client package name
`aerospike-sdk`. It requires Python 3.11+. The released server image name is
`aerospike/aerospike-server`. Required native VECTOR, projection, and Top-K
capabilities must be verified on the actual client/server pair; package names
and version numbers alone are not proof.

Given a caller-populated set and a matching batch embedder:

```python
import dspy
from aerospike_sdk.sync import ClusterDefinition
from dspy.retrievers.aerospike_rm import AerospikeRM

with ClusterDefinition("127.0.0.1", 23000).connect() as cluster:
    rm = AerospikeRM("test", "passages", cluster, embedder, k=3)
    with dspy.context(rm=rm):
        result = dspy.Retrieve(k=2)("What is DSPy?")
    print(result.passages)
```

The local test server configuration is `.devin/aerospike/aerospike.conf`.
It advertises port 23000, uses a single-node replication factor of one, enables
NSUP, and uses a 1 GiB regular file rather than a raw block device.

### Verification

Use the already-prepared environment directly when validating a custom SDK
build: dependency synchronization can replace local editable installations.

```sh
.venv/bin/python -m pytest tests/retrievers/test_aerospike_rm.py -q
DSPY_AEROSPIKE_INTEGRATION=1 AEROSPIKE_PORT=23000 .venv/bin/python -m pytest tests/retrievers -q
.venv/bin/ruff check dspy/retrievers/aerospike_rm.py tests/retrievers/test_aerospike_rm*.py
.venv/bin/ruff format --check dspy/retrievers/aerospike_rm.py tests/retrievers/test_aerospike_rm*.py
.venv/bin/ty check --extra-search-path ../aerospike-client-python-sdk dspy/retrievers/aerospike_rm.py
.venv/bin/ty check
uv lock --check
```

Integration tests are opt-in and use unique sets with exact-key cleanup.
An explicit integration request fails if the server is unreachable or lacks
the required capabilities; it must not be reported as a passing all-skipped run.
`test_dspy_retrieve_context_end_to_end` verifies the actual framework wiring.
`test_more_than_one_thousand_real_results_are_not_truncated` and
`test_concurrent_queries_share_cluster_session_and_preserve_k` cover important
ranking/count and shared-client regressions.

The broader offline CI scope is:

```sh
.venv/bin/python -m pytest tests/ -q -m 'not extra and not deno' -n 2 --dist worksteal
```

Some existing tests import `httpx` directly even when the installed OpenAI
package instead depends on `httpx2`. If collection reports missing `httpx`,
install that test dependency in the environment rather than editing unrelated
tests or host dependency policies.
