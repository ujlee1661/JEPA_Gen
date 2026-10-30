#!/usr/bin/env bash
set -euo pipefail
cd '/mnt/c/Users/God_YJ/Desktop/코드/project_tab_jepa'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root='/home/kangmin/codex_paper_jepa_20261029_3cd171'
task_python="$task_root/runtime/control/bin/python"
task_out="$task_root/retail_a9_combined_v1/online_retail"
mkdir -p "$task_out/orchestration"
echo "combined wave started $(date --iso-8601=seconds)"
"$task_python" -u -m scripts.run_a9_combined_experiment --stage train --arm COMB_FEATURE_ON --device cuda >"$task_out/orchestration/COMB_FEATURE_ON.log" 2>&1 & p1=$!
sleep 20
"$task_python" -u -m scripts.run_a9_combined_experiment --stage train --arm COMB_FEATURE_OFF --device cuda >"$task_out/orchestration/COMB_FEATURE_OFF.log" 2>&1 & p2=$!
if ! wait "$p1"; then wait "$p2" || true; exit 1; fi
wait "$p2"
echo "generation started $(date --iso-8601=seconds)"
"$task_python" -u -m scripts.run_a9_combined_experiment --stage generate --device cuda
"$task_python" -u -m scripts.run_a9_combined_experiment --stage evaluate --device cuda
params=()
for arm in COMB_FEATURE_ON COMB_FEATURE_OFF; do
  for seed in 1 2 3; do
    params+=(--new-sample "$arm=$task_out/generating/$arm/native/$seed/child.csv")
  done
done
for seed in 1 2 3; do
  params+=(--new-sample "A9_ORIGINAL=$task_root/retail_a9_v1/online_retail/generating/A9_FEATURE_ON/native/$seed/child.csv")
done
for seed in 1 2 3; do
  params+=(--new-sample "A9_TIME_ONLY=$task_root/retail_a9_conditioned_time_v1/online_retail/generating/A9_TIME_COND/native/$seed/child.csv")
done
"$task_python" -u -m scripts.report_a7_feature_teacher_comparison --runtime-root "$task_root" --output outputs/paper_reproduction/a9_combined_comparison_v1 "${params[@]}"
echo "complete $(date --iso-8601=seconds)"
