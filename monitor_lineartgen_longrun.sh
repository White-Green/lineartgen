#!/usr/bin/env bash

set -u

PROCESS_NAME="${PROCESS_NAME:-lineartgen}"
INTERVAL="${INTERVAL:-60}"
OUT_BASE="${OUT_BASE:-/tmp}"
REQUESTED_PID="${1:-}"

if [[ "$EUID" -ne 0 ]]; then
  echo "run this monitor with sudo" >&2
  echo "usage: sudo $0 [pid]" >&2
  exit 2
fi

if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERVAL must be a positive integer: $INTERVAL" >&2
  exit 2
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$OUT_BASE/lineartgen-longrun-$TS"
mkdir -p "$OUT_DIR"
chmod 0755 "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$OUT_DIR/run.log"
}

find_amd_device() {
  local card vendor

  for card in /sys/class/drm/card[0-9]*; do
    [[ -r "$card/device/vendor" ]] || continue
    vendor="$(<"$card/device/vendor")"
    if [[ "$vendor" == "0x1002" ]]; then
      readlink -f "$card/device"
      return
    fi
  done
}

resolve_pid() {
  local candidate="$1"
  local tgid

  [[ -r "/proc/$candidate/status" ]] || return 1
  tgid="$(awk '/^Tgid:/ { print $2 }' "/proc/$candidate/status")"
  [[ -n "$tgid" && -r "/proc/$tgid/status" ]] || return 1
  printf '%s\n' "$tgid"
}

wait_for_process() {
  local candidate resolved

  if [[ -n "$REQUESTED_PID" ]]; then
    resolved="$(resolve_pid "$REQUESTED_PID")" || {
      log "process not found: $REQUESTED_PID" >&2
      return 1
    }
    printf '%s\n' "$resolved"
    return
  fi

  log "waiting for process named $PROCESS_NAME" >&2
  while true; do
    candidate="$(pgrep -n -x "$PROCESS_NAME" 2>/dev/null || true)"
    if [[ -n "$candidate" ]]; then
      resolved="$(resolve_pid "$candidate")" || true
      if [[ -n "${resolved:-}" ]]; then
        printf '%s\n' "$resolved"
        return
      fi
    fi
    sleep 1
  done
}

read_status_kib() {
  local key="$1"
  awk -v key="$key" '$1 == key ":" { print $2; found = 1 } END { if (!found) print 0 }' \
    "/proc/$PID/status" 2>/dev/null
}

read_status_value() {
  local key="$1"
  awk -v key="$key" '$1 == key ":" { print $2; found = 1 } END { if (!found) print 0 }' \
    "/proc/$PID/status" 2>/dev/null
}

read_meminfo_kib() {
  local key="$1"
  awk -v key="$key" '$1 == key ":" { print $2; found = 1 } END { if (!found) print 0 }' \
    /proc/meminfo
}

read_device_value() {
  local name="$1"
  local path="$GPU_DEVICE/$name"

  if [[ -r "$path" ]]; then
    tr -d '\n' <"$path"
  else
    printf '0'
  fi
}

debugfs_size() {
  local path="$1"

  if [[ -r "$path" ]]; then
    wc -c <"$path" 2>/dev/null || printf '0'
  else
    printf '0'
  fi
}

drm_fdinfo_values() {
  local files=("/proc/$PID/fdinfo/"*)

  if [[ ! -e "${files[0]}" ]]; then
    printf '0,0,0,0'
    return
  fi

  awk '
    $1 == "drm-memory-vram:" { vram += $2 }
    $1 == "drm-memory-visible-vram:" { visible_vram += $2 }
    $1 == "drm-memory-gtt:" { gtt += $2 }
    $1 == "drm-memory-cpu:" { cpu += $2 }
    END { printf "%.0f,%.0f,%.0f,%.0f", vram, visible_vram, gtt, cpu }
  ' "${files[@]}" 2>/dev/null || printf '0,0,0,0'
}

JOURNAL_PID=""
cleanup() {
  if [[ -n "$JOURNAL_PID" ]]; then
    kill "$JOURNAL_PID" 2>/dev/null || true
    wait "$JOURNAL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

log "output: $OUT_DIR"

if command -v journalctl >/dev/null 2>&1; then
  journalctl -k -f -o short-iso --since now >"$OUT_DIR/kernel.log" 2>&1 &
  JOURNAL_PID=$!
else
  echo "journalctl not found" >"$OUT_DIR/kernel.log"
fi

PID="$(wait_for_process)" || exit 2
GPU_DEVICE="$(find_amd_device || true)"
START_SECONDS="$SECONDS"

log "monitoring pid $PID every ${INTERVAL}s"
log "AMD device: ${GPU_DEVICE:-not found}"

{
  echo "date: $(date --iso-8601=seconds)"
  echo "hostname: $(hostname)"
  echo "kernel: $(uname -a)"
  echo "pid: $PID"
  echo "command: $(tr '\0' ' ' <"/proc/$PID/cmdline")"
  echo "gpu_device: ${GPU_DEVICE:-not found}"
  echo
  cat "/proc/$PID/limits"
} >"$OUT_DIR/process-info.txt" 2>&1

cat >"$OUT_DIR/samples.csv" <<'EOF'
timestamp,elapsed_seconds,pid,threads,fd_count,map_count,vm_size_kib,vm_rss_kib,rss_anon_kib,rss_file_kib,rss_shmem_kib,vm_swap_kib,vm_pte_kib,system_mem_available_kib,system_swap_free_kib,gpu_busy_percent,vram_used_bytes,vram_total_bytes,visible_vram_used_bytes,gtt_used_bytes,gtt_total_bytes,preempt_used_bytes,process_drm_vram_kib,process_drm_visible_vram_kib,process_drm_gtt_kib,process_drm_cpu_kib,debug_vm_info_bytes,debug_gem_info_bytes
EOF

while [[ -r "/proc/$PID/status" ]]; do
  NOW="$(date --iso-8601=seconds)"
  ELAPSED="$((SECONDS - START_SECONDS))"
  FD_COUNT="$(find "/proc/$PID/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
  MAP_COUNT="$(wc -l <"/proc/$PID/maps" 2>/dev/null || printf '0')"
  DRM_VALUES="$(drm_fdinfo_values)"
  IFS=',' read -r DRM_VRAM_KIB DRM_VISIBLE_VRAM_KIB DRM_GTT_KIB DRM_CPU_KIB \
    <<<"$DRM_VALUES"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$NOW" \
    "$ELAPSED" \
    "$PID" \
    "$(read_status_value Threads)" \
    "$FD_COUNT" \
    "$MAP_COUNT" \
    "$(read_status_kib VmSize)" \
    "$(read_status_kib VmRSS)" \
    "$(read_status_kib RssAnon)" \
    "$(read_status_kib RssFile)" \
    "$(read_status_kib RssShmem)" \
    "$(read_status_kib VmSwap)" \
    "$(read_status_kib VmPTE)" \
    "$(read_meminfo_kib MemAvailable)" \
    "$(read_meminfo_kib SwapFree)" \
    "$(read_device_value gpu_busy_percent)" \
    "$(read_device_value mem_info_vram_used)" \
    "$(read_device_value mem_info_vram_total)" \
    "$(read_device_value mem_info_vis_vram_used)" \
    "$(read_device_value mem_info_gtt_used)" \
    "$(read_device_value mem_info_gtt_total)" \
    "$(read_device_value mem_info_preempt_used)" \
    "$DRM_VRAM_KIB" \
    "$DRM_VISIBLE_VRAM_KIB" \
    "$DRM_GTT_KIB" \
    "$DRM_CPU_KIB" \
    "$(debugfs_size /sys/kernel/debug/dri/0/amdgpu_vm_info)" \
    "$(debugfs_size /sys/kernel/debug/dri/0/amdgpu_gem_info)" \
    >>"$OUT_DIR/samples.csv"

  {
    echo "===== $NOW ====="
    if command -v amd-smi >/dev/null 2>&1; then
      timeout 15 amd-smi metric -m -u --csv || true
      timeout 15 amd-smi process -G -e -p "$PID" --csv || true
    else
      echo "amd-smi not found"
    fi
  } >>"$OUT_DIR/amd-smi.txt" 2>&1

  sleep "$INTERVAL" &
  wait $! 2>/dev/null || true
done

log "process $PID exited"

if command -v journalctl >/dev/null 2>&1; then
  journalctl -k -o short-iso --since "10 minutes ago" >"$OUT_DIR/kernel-final-10m.log" 2>&1 || true
fi

cleanup
JOURNAL_PID=""

TAR_PATH="$OUT_DIR.tar.gz"
tar -C "$(dirname "$OUT_DIR")" -czf "$TAR_PATH" "$(basename "$OUT_DIR")" || true
chmod a+r "$TAR_PATH" 2>/dev/null || true

log "done"
log "directory: $OUT_DIR"
log "archive: $TAR_PATH"
