#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

real_ranking_json="${GHOST_MGG_M4_REAL_RANKING_JSON:-${repo_root}/reports/m5_real_d435_ranking/m5_real_ranking.json}"
graspability_json="${GHOST_MGG_M4_GRASPABILITY_JSON:-${repo_root}/reports/m4_graspability_dryrun/graspability.json}"
moveit_json="${GHOST_MGG_M4_MOVEIT_JSON:-${repo_root}/reports/m4_sim_moveit_dryrun/plan_results.json}"
output_dir="${GHOST_MGG_M4_JOINT_OUTPUT_DIR:-${repo_root}/reports/m4_joint_hypotheses}"

if [[ ! -f "${real_ranking_json}" ]]; then
  echo "Missing real ranking report: ${real_ranking_json}" >&2
  exit 2
fi
if [[ ! -f "${graspability_json}" ]]; then
  echo "Missing graspability report: ${graspability_json}" >&2
  exit 2
fi
if [[ ! -f "${moveit_json}" ]]; then
  "${script_dir}/smoke_m4_sim_grasp_moveit_dryrun.sh"
fi

export PYTHONPATH="${repo_root}/src/ghost_mgg_core/python:${PYTHONPATH:-}"
python3 -m ghost_mgg_core_py.evaluation.m4_joint_hypothesis_report \
  --real-ranking-json "${real_ranking_json}" \
  --graspability-json "${graspability_json}" \
  --moveit-json "${moveit_json}" \
  --output-dir "${output_dir}"
