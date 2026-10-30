# Native sequence evaluation

`jets_multidata_report.py` is a read-only descriptive comparison of native
generation outputs against a common `account.csv`/`trans.csv` reference. It
records per-generation values, averages generations **within each independent
training seed**, and reports mean and sample standard deviation across
training seeds separately. A three-generation average from one checkpoint is
not an estimate of training-seed stability.

## JETS case manifest

The JETS study layout contains `DATASET/train_SEED/generation_SEED/{child.csv,counts.csv,metrics.json}`.
Create the 36 planned cases without inspecting or modifying the study:

```bash
python -m analysis.collect_jets_cases --study-root /path/to/study --output /path/to/new_cases.json
python -m analysis.jets_multidata_report --reference-root /path/to/reference/data \
  --cases /path/to/new_cases.json --output /path/to/new_report.json --allow-partial
```

`--allow-partial` records unavailable cases rather than silently counting them
as completed. JETS cases require the completed `generation.json` and `counts.csv`
receipts before a row is considered available; child digest and seed are checked.
Each JETS row also links the training checkpoint, completion receipt and manifest,
checking checkpoint digest, update count, training seed and generation checkpoint.
Coverage completion and verified training provenance are reported separately.
For JETS the pipeline receipt additionally verifies the original input hashes,
4096 windows, snapshot seed and budgets, and both target encoder manifests,
training completion, training-only calibration, model and suffix-cache SHA chain.
The generator manifest must record those exact artifacts, the feature prediction
loss ON, deterministic conditioning, and no actual-future generation inputs.
Within each training seed, all three generation receipts must name the same
generator checkpoint. A missing or mismatched link is not a verified result.
Each report path must be new; the source tables, receipts and generated outputs
are not modified. The report records input hashes.

## Other models

Add native paths to `samples` in the manifest, with explicit `dataset`, `model`,
`training_seed`, `generation_seed`, `child` and `count_source`. The count
source must say whether N was generated or supplied: equal N is not evidence
of free-count generation. `counts` and `metrics` paths are optional, though a
complete five-score comparison requires pinned original evaluator receipts.
Optional `checkpoint`, `training_receipt`, `training_manifest` must be supplied
together to use the generic checkpoint/seed verification; otherwise the result
is explicitly marked as having unverified training provenance. For TabARGN,
`training_evidence` can instead point to its `training_complete.json`: the
training seed, exit status, declared budget and every listed weight digest are
verified. A matching training receipt by itself does **not** prove that a
generated CSV was produced by those weights; generation-to-checkpoint linkage
is tracked separately. Other baseline receipt formats require separate
verification and must not be represented as independently checked here.

For the supported TabARGN study layout, create planned cases with:

```bash
python -m analysis.collect_tabargn_cases --study-root /path/to/baseline_study \
  --output /path/to/new_tabargn_cases.json
```

These cases wait for `verified_samples`, then require a bitwise match with the
original child table and equality of the before/after checkpoint hashes with
the completed training receipt. The original count receipt is independently
checked against the replayed rows. Missing verification is reported as pending.

ClavaDDPM has synthetic parents and a different completion-receipt format:

```bash
python -m analysis.collect_clava_cases --study-root /path/to/baseline_study \
  --output /path/to/new_clava_cases.json
```

For ClavaDDPM, the reporter checks generated parent/child/count hashes, the
link to the training receipt and the recorded model-file hashes. Its current
generation receipt does not include direct before/after model hashes, and its
training completion receipt has no explicit exit-code field; both remain
separately marked unverified rather than inferred.
When present, counts must cover every reference parent and match the generated
rows. A receipt lacking a child digest is explicitly marked as not directly
linked; an incorrect evaluator digest is rejected. For models conditioning on
the fixed reference parents, set `parent_identity` to `reference` (the
default), and ID membership is checked. For models that synthesize parents,
set `parent_identity` to `generated` and provide `generated_parents` with its
`generated_parent_column`, or provide a complete `counts` receipt: these IDs
are compared as a separate population and are never mapped to real IDs. If
neither generated parents nor a complete counts receipt is available, empty
generated parents cannot be inferred; all-parent N KS is unavailable and only
the explicitly labeled non-empty-parent N KS is reported.
For a count file with non-default column names, set `counts_id_column` and
`counts_value_column` explicitly (for example, `entity` and `N` for an original
baseline count receipt).

For each replicate, the report contains KS distances for N (including empty
parents), internal event gaps, and non-empty active spans; source-scale
numerical summaries, per-row `expm1` for `log_` fields, and observed categorical
total variation and missing fractions. It does not infer whether a `log_`
column is monetary; for example, `log_page` describes pages rather than money.
Pooled marginal agreement does not prove customer-level or time-feature
dependence. Full-population fidelity is not held-out generalization.
# Remaining-baseline receipts

For RGCLD or RelDiff, set `training_evidence_profile` to `remaining`, point
`training_evidence` and `receipt` at the original training and generation
receipts, and set `job`, `checkpoint_root` (the job's sandbox), and
`reference_baseline_root` (the corresponding reference baseline directory).
Set `training_budget_field`, `expected_training_budget`, and
`expected_job_budget` to the actual intended budgets, `expected_count_regime`
to the generation receipt's documented count regime, and require the weight
link. The reporter checks the recorded checkpoint map, original and sandbox
source and inputs, seeds, and generated artifact digests. A checkpoint link
established by the generation receipt does not itself prove a separate
before-and-after checkpoint measurement; missing exit codes remain unverified.
