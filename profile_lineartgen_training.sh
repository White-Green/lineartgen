#!/usr/bin/env bash
set -u

PID="${1:-}"
DURATION="${DURATION:-30}"
PERF_FREQ="${PERF_FREQ:-99}"
OUT_BASE="${OUT_BASE:-/tmp}"

if [[ -z "$PID" ]]; then
  echo "usage: sudo $0 <pid>" >&2
  exit 2
fi

INPUT_PID="$PID"
if ! ps -p "$PID" >/dev/null 2>&1; then
  if [[ -r "/proc/$PID/status" ]]; then
    TGID="$(awk '/^Tgid:/ { print $2 }' "/proc/$PID/status")"
    if [[ -n "${TGID:-}" ]] && ps -p "$TGID" >/dev/null 2>&1; then
      PID="$TGID"
    fi
  fi
fi

if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "process not found: $INPUT_PID" >&2
  echo "hint: if this is an htop thread id, try: cat /proc/$INPUT_PID/status | grep -E '^(Name|Pid|Tgid|PPid):'" >&2
  exit 2
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$OUT_BASE/lineartgen-profile-$PID-$TS"
mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT_DIR/run.log"
}

find_perf() {
  local kernel_perf="/usr/lib/linux-tools/$(uname -r)/perf"
  if [[ -x "$kernel_perf" ]]; then
    echo "$kernel_perf"
    return
  fi

  local direct_perf
  direct_perf="$(find /usr/lib -type f -path '*linux-tools*' -name perf 2>/dev/null | sort -Vr | head -n 1)"
  if [[ -n "$direct_perf" && -x "$direct_perf" ]]; then
    echo "$direct_perf"
    return
  fi

  command -v perf || true
}

log "output: $OUT_DIR"
log "input pid/tid: $INPUT_PID"
log "process pid: $PID"
log "duration: ${DURATION}s"

{
  echo "date: $(date --iso-8601=seconds)"
  echo "hostname: $(hostname)"
  echo "kernel: $(uname -a)"
  echo "cwd: $(pwd)"
  echo "input pid/tid: $INPUT_PID"
  echo "process pid: $PID"
  echo
  echo "process:"
  ps -fp "$PID" || true
  echo
  echo "threads:"
  ps -L -p "$PID" -o pid,tid,psr,pcpu,stat,wchan:32,comm --sort=-pcpu || true
  echo
  echo "status:"
  cat "/proc/$PID/status" 2>/dev/null || true
  echo
  echo "limits:"
  cat "/proc/$PID/limits" 2>/dev/null || true
} >"$OUT_DIR/process-info.txt" 2>&1

log "phase 1: lightweight CPU/GPU monitoring"

if command -v pidstat >/dev/null 2>&1; then
  pidstat -t -h -p "$PID" 1 "$DURATION" >"$OUT_DIR/pidstat.txt" 2>&1 &
  PIDSTAT_PID=$!
else
  echo "pidstat not found" >"$OUT_DIR/pidstat.txt"
  PIDSTAT_PID=""
fi

if command -v amd-smi >/dev/null 2>&1; then
  amd-smi monitor -u -m -v -p -q -w 1 -i "$DURATION" --csv --file "$OUT_DIR/amd-smi.csv" \
    >"$OUT_DIR/amd-smi.stdout.txt" 2>"$OUT_DIR/amd-smi.stderr.txt" &
  AMD_SMI_PID=$!
else
  echo "amd-smi not found" >"$OUT_DIR/amd-smi.stderr.txt"
  AMD_SMI_PID=""
fi

if [[ -n "${PIDSTAT_PID:-}" ]]; then
  wait "$PIDSTAT_PID" 2>/dev/null || true
fi
if [[ -n "${AMD_SMI_PID:-}" ]]; then
  wait "$AMD_SMI_PID" 2>/dev/null || true
fi

ps -L -p "$PID" -o pid,tid,psr,pcpu,stat,wchan:32,comm --sort=-pcpu \
  >"$OUT_DIR/threads-after-monitor.txt" 2>&1 || true

log "phase 2: syscall summary with strace"
if command -v strace >/dev/null 2>&1; then
  timeout "$DURATION" strace -f -c -p "$PID" -o "$OUT_DIR/strace-summary.txt" \
    >"$OUT_DIR/strace.stdout.txt" 2>"$OUT_DIR/strace.stderr.txt" || true
else
  echo "strace not found" >"$OUT_DIR/strace-summary.txt"
fi

log "phase 3: perf sampling"
PERF_BIN="$(find_perf)"
echo "$PERF_BIN" >"$OUT_DIR/perf-bin.txt"

PERF_OK=0
if [[ -n "$PERF_BIN" && -x "$PERF_BIN" ]]; then
  "$PERF_BIN" record \
    -o "$OUT_DIR/perf.data" \
    -F "$PERF_FREQ" \
    -g --call-graph dwarf \
    -p "$PID" -- sleep "$DURATION" \
    >"$OUT_DIR/perf-record.stdout.txt" 2>"$OUT_DIR/perf-record.stderr.txt" && PERF_OK=1

  if [[ "$PERF_OK" -eq 1 ]]; then
    "$PERF_BIN" report \
      -i "$OUT_DIR/perf.data" \
      --stdio --no-children \
      --sort comm,dso,symbol \
      >"$OUT_DIR/perf-report.txt" 2>"$OUT_DIR/perf-report.stderr.txt" || true

    "$PERF_BIN" script \
      -i "$OUT_DIR/perf.data" \
      >"$OUT_DIR/perf-script.txt" 2>"$OUT_DIR/perf-script.stderr.txt" || true

    gzip -f "$OUT_DIR/perf-script.txt" 2>/dev/null || true
  fi
else
  echo "perf not found" >"$OUT_DIR/perf-record.stderr.txt"
fi

if [[ "$PERF_OK" -ne 1 ]]; then
  log "perf failed; collecting gdb stack samples"
  if command -v gdb >/dev/null 2>&1; then
    : >"$OUT_DIR/gdb-stacks.txt"
    for i in $(seq 1 "$DURATION"); do
      {
        echo "===== sample $i $(date --iso-8601=seconds) ====="
        gdb -batch -p "$PID" -ex "thread apply all bt 10"
      } >>"$OUT_DIR/gdb-stacks.txt" 2>&1
      sleep 1
      ps -p "$PID" >/dev/null 2>&1 || break
    done
  else
    echo "gdb not found" >"$OUT_DIR/gdb-stacks.txt"
  fi
fi

{
  echo "Generated files:"
  find "$OUT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
  echo
  echo "Useful first reads:"
  echo "- process-info.txt"
  echo "- pidstat.txt"
  echo "- amd-smi.csv"
  echo "- strace-summary.txt"
  echo "- perf-report.txt, if perf succeeded"
  echo "- gdb-stacks.txt, if perf failed"
} >"$OUT_DIR/README.txt"

TAR_PATH="$OUT_DIR.tar.gz"
tar -C "$(dirname "$OUT_DIR")" -czf "$TAR_PATH" "$(basename "$OUT_DIR")" \
  >"$OUT_DIR/tar.stdout.txt" 2>"$OUT_DIR/tar.stderr.txt" || true

log "done"
log "directory: $OUT_DIR"
log "archive: $TAR_PATH"

echo "$OUT_DIR"
echo "$TAR_PATH"
