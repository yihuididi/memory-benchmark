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

# RippleEdits source attribution

Source: https://github.com/edenbiran/RippleEdits
Revision: `54f3b88af4895a3aacb580ec63ce7ae857185040`
Copyright (c) 2023 Eden Biran.
License: MIT (see `THIRD_PARTY_LICENSES/RIPPLE_EDITS_LICENSE.txt`).

`src/memory_bench/benchmarks/ripple_edit.py` adapts the dataset structure and
all-entities/any-alias and grouped-test evaluation behavior documented in
upstream `query.py`, `queryexecutor.py`, and `testrunner.py`. It implements a
memory benchmark without the upstream model-editing or Wikidata runtime.
It does not perform pre-edit knowledge checks or edit-success gating. Matching
uses normalized text and word boundaries, and unscorable queries are excluded
instead of automatically passing. These scores are not directly comparable
to the original paper's protocol.

Reference: Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva (2024),
“Evaluating the Ripple Effects of Knowledge Editing in Language Models,”
Transactions of the Association for Computational Linguistics 12:283–298.
