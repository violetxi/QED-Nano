#!/bin/bash
# Monitor pipelinerl training and auto-restart on failure.
# Usage: bash monitor_and_restart.sh
#
# Detects failure by:
#   1. Main bash process exiting (set -e kills it on any error)
#   2. Rapid error bursts in the actor error log (>50 errors in 60s)
#   3. All GPU utilization dropping to 0% for extended period (stalled)
#
# On failure: kills all related processes, waits for GPUs to free, restarts.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$SCRIPT_DIR"
LAUNCH_SCRIPT="run_dense_process.sh"
ERROR_LOG="$TRAINING_DIR/results/dense_process/actor/error.log"
LAUNCH_LOG="$TRAINING_DIR/results/dense_process/launch.log"
MONITOR_LOG="$TRAINING_DIR/monitor.log"
CHECK_INTERVAL=60        # seconds between checks
ERROR_THRESHOLD=50       # errors in last CHECK_INTERVAL seconds to trigger restart
MAX_RESTARTS=20          # safety: don't restart forever
RESTART_COUNT=0
GPU_CLEAR_TIMEOUT=120    # seconds to wait for GPUs to free

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

find_main_pid() {
    # Find the bash process running run_dense_process.sh
    pgrep -f "bash ${LAUNCH_SCRIPT}" 2>/dev/null | head -1
}

count_recent_errors() {
    # Count error lines in the last CHECK_INTERVAL seconds
    if [ ! -f "$ERROR_LOG" ]; then
        echo 0
        return
    fi
    local cutoff
    cutoff=$(date -d "-${CHECK_INTERVAL} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
    if [ -z "$cutoff" ]; then
        echo 0
        return
    fi
    # Count lines with timestamps after the cutoff
    awk -v cutoff="$cutoff" '
    match($0, /[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/, ts) {
        if (ts[0] >= cutoff) count++
    }
    END { print count+0 }
    ' "$ERROR_LOG"
}

check_gpu_processes() {
    # Check if there are python processes using GPUs (pipelinerl-related)
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'
}

kill_all_training() {
    log "Killing all training processes..."

    # Kill the main bash process and its children
    local main_pid
    main_pid=$(find_main_pid)
    if [ -n "$main_pid" ]; then
        log "Killing main process tree (PID: $main_pid)"
        kill -- -"$(ps -o pgid= -p "$main_pid" | tr -d ' ')" 2>/dev/null || true
        sleep 2
        kill -9 -- -"$(ps -o pgid= -p "$main_pid" | tr -d ' ')" 2>/dev/null || true
    fi

    # Kill any lingering pipelinerl processes
    pkill -f "pipelinerl.entrypoints" 2>/dev/null || true
    pkill -f "pipelinerl.launch" 2>/dev/null || true
    pkill -f "accelerate.commands.launch.*run_finetune" 2>/dev/null || true
    sleep 3
    pkill -9 -f "pipelinerl.entrypoints" 2>/dev/null || true
    pkill -9 -f "pipelinerl.launch" 2>/dev/null || true
    pkill -9 -f "accelerate.commands.launch.*run_finetune" 2>/dev/null || true

    # Wait for GPUs to clear
    log "Waiting for GPUs to free..."
    local waited=0
    while [ "$waited" -lt "$GPU_CLEAR_TIMEOUT" ]; do
        local gpu_procs
        gpu_procs=$(check_gpu_processes)
        if [ "$gpu_procs" -eq 0 ]; then
            log "GPUs are clear."
            return 0
        fi
        log "Still $gpu_procs GPU processes running, waiting... ($waited/${GPU_CLEAR_TIMEOUT}s)"
        sleep 5
        waited=$((waited + 5))
    done

    # Force kill any remaining GPU processes
    log "Timeout waiting for GPUs. Force killing remaining GPU processes."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 5
}

start_training() {
    log "Starting training (attempt $((RESTART_COUNT + 1)))..."
    cd "$TRAINING_DIR"

    # Record error log size before start so we don't re-count old errors
    if [ -f "$ERROR_LOG" ]; then
        ERROR_LOG_BASELINE=$(wc -l < "$ERROR_LOG")
    else
        ERROR_LOG_BASELINE=0
    fi

    # Launch in background within a new process group
    setsid bash "$LAUNCH_SCRIPT" >> "$MONITOR_LOG" 2>&1 &
    TRAINING_PGID=$!

    # Give it time to spawn
    sleep 15

    local main_pid
    main_pid=$(find_main_pid)
    if [ -n "$main_pid" ]; then
        log "Training started with main PID: $main_pid"
        return 0
    else
        log "WARNING: Could not find main process after start"
        return 1
    fi
}

is_training_healthy() {
    # Check 1: Is the main process still alive?
    local main_pid
    main_pid=$(find_main_pid)
    if [ -z "$main_pid" ]; then
        log "FAILURE DETECTED: Main process is no longer running"
        return 1
    fi

    # Check 2: Are there recent error bursts?
    local recent_errors
    recent_errors=$(count_recent_errors)
    if [ "$recent_errors" -ge "$ERROR_THRESHOLD" ]; then
        log "FAILURE DETECTED: $recent_errors errors in the last ${CHECK_INTERVAL}s (threshold: $ERROR_THRESHOLD)"
        return 1
    fi

    # Check 3: Are GPU processes still there?
    local gpu_procs
    gpu_procs=$(check_gpu_processes)
    if [ "$gpu_procs" -lt 2 ]; then
        # At minimum we expect finetune (2 GPUs) + actors, but even 2 is a sign of life
        log "FAILURE DETECTED: Only $gpu_procs GPU processes (expected 8+)"
        return 1
    fi

    return 0
}

# ---- Main loop ----

log "========================================="
log "Training monitor started"
log "Script: $LAUNCH_SCRIPT"
log "Check interval: ${CHECK_INTERVAL}s"
log "Error threshold: $ERROR_THRESHOLD errors/${CHECK_INTERVAL}s"
log "Max restarts: $MAX_RESTARTS"
log "========================================="

# Check if training is already running
EXISTING_PID=$(find_main_pid)
if [ -n "$EXISTING_PID" ]; then
    log "Training is already running (PID: $EXISTING_PID). Monitoring it."
else
    log "No training running. Starting fresh."
    start_training
    RESTART_COUNT=$((RESTART_COUNT + 1))
fi

while true; do
    sleep "$CHECK_INTERVAL"

    if is_training_healthy; then
        # Periodically log that things are OK
        local_errors=$(count_recent_errors)
        log "OK - PID: $(find_main_pid), GPU procs: $(check_gpu_processes), recent errors: $local_errors"
    else
        if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
            log "ABORT: Reached max restart limit ($MAX_RESTARTS). Manual intervention needed."
            exit 1
        fi

        log "Initiating restart ($((RESTART_COUNT + 1))/$MAX_RESTARTS)..."
        kill_all_training
        sleep 10  # cooldown before restart
        start_training
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "Restart complete. Resuming monitoring."
    fi
done
