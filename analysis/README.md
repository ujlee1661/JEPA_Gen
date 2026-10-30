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
as completed. Each report path must be new; the source tables, receipts and
generated outputs are not modified. The report records input hashes.

## Other models

Add native paths to `samples` in the manifest, with explicit `dataset`, `model`,
`training_seed`, `generation_seed`, `child` and `count_source`. The count
source must say whether N was generated or supplied: equal N is not evidence
of free-count generation. `counts` and `metrics` paths are optional, though a
complete five-score comparison requires pinned original evaluator receipts.
When present, counts must cover every reference parent and match the generated
rows. A receipt lacking a child digest is explicitly marked as not directly
linked; an incorrect evaluator digest is rejected. IDs must correspond to the
reference population; this tool does not relabel or repair them.

For each replicate, the report contains KS distances for N (including empty
parents), internal event gaps, and non-empty active spans; source-scale
numerical summaries, per-row `expm1` for `log_` fields, and observed categorical
total variation and missing fractions. It does not infer whether a `log_`
column is monetary; for example, `log_page` describes pages rather than money.
Pooled marginal agreement does not prove customer-level or time-feature
dependence. Full-population fidelity is not held-out generalization.
