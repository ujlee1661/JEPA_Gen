# A7 feature-teacher / baseline comparison

Full417 same-parent IN-SAMPLE fidelity; no heldout generalization claim.

All available predeclared generation seeds are retained. Each row averages generations within ONE training seed. Teacher-oracle reconstruction is separate from native generation; pending new paths are not scored as zero.

| Model / training seed | Gen seeds | TS | CS | SD | LG | Joint FF | Count source |
|---|---|---:|---:|---:|---:|---:|---|
| A7_ENERGY / 9 | [1, 2, 3] | 0.93493 | 0.70240 | 0.92438 | 0.89920 | 0.89289 | Generated-N native; all417 static parents; empty paths retained in counts.csv |
| A7_POINT_OFF / 9 | [1, 2, 3] | 0.92407 | 0.66923 | 0.85110 | 0.91096 | 0.88889 | Generated-N native; all417 static parents; empty paths retained in counts.csv |
| A7_POINT_ON / 9 | [1, 2, 3] | 0.93405 | 0.69114 | 0.91422 | 0.91067 | 0.91047 | Generated-N native; all417 static parents; empty paths retained in counts.csv |
| A9_ON / 9 | [1, 2, 3] | 0.94261 | 0.68957 | 0.90444 | 0.90834 | 0.90328 | Generated-N native; all417 static parents; empty paths retained in counts.csv |
| A9_TIME_COND / 9 | [1, 2, 3] | 0.92570 | 0.69447 | 0.92332 | 0.90583 | 0.89848 | Generated-N native; all417 static parents; empty paths retained in counts.csv |
| RDBDiff_GivenN_graph / 1 | [1] | 0.78697 | 0.71651 | 0.99412 | 0.85956 | 0.78177 | Given real N + fixed structure; NOT count-free |
| TabARGN / 9 | [1, 2, 3] | 0.85495 | 0.65061 | 0.90679 | 0.87774 | 0.84412 | Generated-N |

| Model / training seed | Amount mean | Amount std | Amount P99 | Zero % | Customer mean SD | Cum std | Cum decrease % | Δcum−amount MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| REAL | 496.89896 | 1258.75591 | 5091.37240 | 19.59637 | 459.98239 | 56128.60440 | 0.00000 | 0.00000 |
| A7_ENERGY / 9 | 341.91284 | 618.00751 | 2333.19179 | 28.63283 | 205.94825 | 27117.20978 | 27.86948 | 603.08432 |
| A7_POINT_OFF / 9 | 254.73575 | 419.02211 | 1422.06595 | 26.33200 | 162.11781 | 9853.84923 | 24.43618 | 221.32782 |
| A7_POINT_ON / 9 | 264.49684 | 366.30493 | 1405.47461 | 26.04449 | 135.55555 | 28503.61079 | 25.58554 | 704.08074 |
| A9_ON / 9 | 289.03053 | 497.02862 | 1782.73262 | 30.83251 | 149.28799 | 27342.82154 | 26.01793 | 992.99987 |
| A9_TIME_COND / 9 | 324.67531 | 613.17631 | 2334.53604 | 31.59018 | 175.35189 | 33246.39510 | 25.63430 | 878.78005 |
| RDBDiff_GivenN_graph / 1 | 488.97086 | 1322.76807 | 4820.28067 | 19.79326 | 296.35668 | 55114.00427 | 47.83120 | 20540.04853 |
| TabARGN / 9 | 418.43167 | 1048.05194 | 3772.55225 | 16.71422 | 199.62865 | 31005.18746 | 39.66404 | 7808.80324 |

N/time distances, every individual seed and original log-column statistics are retained in CSV/JSON. All source-account IDs remain in the N distribution including empty generated parents. Cum decreases are descriptive; this Retail increment test is evaluated with the same source floating baseline, never enforced in a generator.

## Pending or excluded inputs


## Limits

- Generation SD is not training-seed uncertainty; no best seed selected.
- RDBDiff Given-N/fixed-graph row is not evidence of count-free generation.
- Existing A7_POINT_OFF is suffix auxiliary-off, not a verified complete no-JEPA pipeline.
- Different legacy training budgets/front/teacher reuse costs are not controlled by this descriptive table.
- New outputs are scored only after fixed-final provenance; pending/oracle/forecast cases are shown separately.
- Native generation artifacts and oracle teacher reconstruction analyses must not be pooled.

[Physical-distribution dashboard](comparison.html)
