#!/usr/bin/env bash
set -euo pipefail

tmp_root="${TMPDIR:-/tmp}"
log_file="$(mktemp "${tmp_root%/}/ghost_mgg_d435_realsense.XXXXXX.log")"
pid_file="$(mktemp "${tmp_root%/}/ghost_mgg_d435_realsense.XXXXXX.pid")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root=""
launch_pid=""
launch_pgid=""
setsid_pid=""

find_repo_root() {
  local candidate="${script_dir}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -d "${candidate}/.git" && -d "${candidate}/src/ghost_mgg_real" ]]; then
      repo_root="${candidate}"
      return 0
    fi
    candidate="$(dirname "${candidate}")"
  done

  if [[ -n "${GHOST_MGG_WS:-}" && -d "${GHOST_MGG_WS}/src/ghost_mgg_real" ]]; then
    repo_root="${GHOST_MGG_WS}"
    return 0
  fi

  if [[ -d "${PWD}/src/ghost_mgg_real" ]]; then
    repo_root="${PWD}"
    return 0
  fi

  echo "could not locate Ghost-MGG workspace root; run from the repo or set GHOST_MGG_WS" >&2
  return 1
}

cleanup() {
  set +e
  if [[ -n "${launch_pgid}" ]] && kill -0 "-${launch_pgid}" 2>/dev/null; then
    kill -INT "-${launch_pgid}" 2>/dev/null
    for _ in {1..8}; do
      kill -0 "-${launch_pgid}" 2>/dev/null || break
      sleep 1
    done
    kill -TERM "-${launch_pgid}" 2>/dev/null
    wait "${launch_pid}" 2>/dev/null
    if [[ -n "${setsid_pid}" && "${setsid_pid}" != "${launch_pid}" ]]; then
      wait "${setsid_pid}" 2>/dev/null
    fi
  fi
  rm -f "${pid_file}"
}

trap cleanup EXIT

set +u
source /opt/ros/jazzy/setup.bash
set -u
find_repo_root
set +u
source "${repo_root}/install/setup.bash"
set -u

cd "${repo_root}"
ros2 daemon stop >/dev/null 2>&1 || true

setsid bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' bash \
  "${pid_file}" ros2 launch ghost_mgg_real d435_realsense.launch.py \
  >"${log_file}" 2>&1 &
setsid_pid=$!

for _ in {1..30}; do
  if [[ -s "${pid_file}" ]]; then
    launch_pid="$(<"${pid_file}")"
    launch_pgid="${launch_pid}"
    break
  fi
  sleep 0.1
done

if [[ -z "${launch_pid}" ]]; then
  launch_pid="${setsid_pid}"
  launch_pgid="${setsid_pid}"
fi

deadline=$((SECONDS + 45))

topic_once() {
  timeout 8s ros2 topic echo --once "$@" >/dev/null 2>&1
}

topic_field_contains() {
  local topic="$1"
  local field="$2"
  local expected="$3"
  timeout 8s ros2 topic echo --once --field "${field}" "${topic}" 2>/dev/null |
    grep -Fxq "${expected}"
}

while (( SECONDS < deadline )); do
  if grep -Fq "Device USB type: 3." "${log_file}" &&
    grep -Fq "RealSense Node Is Up!" "${log_file}" &&
    topic_once /camera/camera/depth/camera_info &&
    topic_once /camera/camera/color/camera_info &&
    topic_field_contains /camera/camera/depth/image_rect_raw encoding 16UC1 &&
    topic_field_contains /camera/camera/color/image_raw encoding rgb8 &&
    topic_field_contains /camera/camera/infra1/image_rect_raw encoding mono8 &&
    topic_field_contains /camera/camera/depth/color/points header.frame_id camera_depth_optical_frame
  then
    echo "D435 Realsense smoke passed"
    exit 0
  fi
  sleep 1
done

echo "D435 Realsense smoke failed" >&2
echo "---- launch log tail (${log_file}) ----" >&2
tail -n 160 "${log_file}" >&2 || true
echo "---- camera topics ----" >&2
ros2 topic list 2>/dev/null | sort | grep -E '^/camera|^/tf' >&2 || true
exit 1
