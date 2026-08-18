#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export PYTHONPATH="${repo_root}/src/ghost_mgg_core/python:${PYTHONPATH:-}"
python3 -m ghost_mgg_core_py.evaluation.m4_no_truth_gate \
  --repo-root "${repo_root}" \
  --real-dashboard-json "${repo_root}/reports/m4_real_dashboard/dashboard.json" \
  --multi-target-json "${repo_root}/reports/m4_no_truth_multi_target/multi_target.json" \
  --scenario-sweep-json "${repo_root}/reports/m4_no_truth_scenario_sweep/scenario_sweep.json" \
  --dynamic-execute-json "${repo_root}/reports/m4_no_truth_live_dynamic_execute/result.json" \
  --ranked-fallback-json "${repo_root}/reports/m4_no_truth_live_ranked_fallback/result.json" \
  --output-dir "${repo_root}/reports/m4_no_truth_gate"
