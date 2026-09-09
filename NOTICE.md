# LongMemEval-V2 source attribution

Source: https://github.com/xiaowu0162/LongMemEval-V2
Revision: `2cc8c540bdb87fe6761629b585e727e1c4704520`
License: Apache-2.0 (see `third_party_licenses/LONGMEMEVAL_V2_LICENSE`).

`qa_eval_metrics.py` is copied without modifications from evaluation/qa_eval_metrics.py.
`harness.py` contains selected functions and constants from evaluation/harness.py,
with standalone imports and a local context-item type alias. Token counting and truncation accept an optional processor argument so callers can
use a local, revision-pinned processor without changing global state. Other function
bodies and all prompt literals are preserved verbatim, including upstream escape sequences.
The surrounding integration supplies optional dependencies, configuration,
resource lifecycle, and flattened numeric metrics for this repository.
