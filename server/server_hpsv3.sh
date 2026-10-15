#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for tool in python ray timeout realpath; do
    command -v "$tool" >/dev/null || { printf 'Missing command: %s\n' "$tool" >&2; exit 1; }
done
: "${HPSV3_CHECKPOINT_PATH:?Set HPSV3_CHECKPOINT_PATH to HPSv3.safetensors}"
: "${HPSV3_BASE_MODEL_PATH:?Set HPSV3_BASE_MODEL_PATH to a local Qwen2-VL-7B-Instruct directory}"
export HPSV3_CHECKPOINT_PATH="$(realpath -e "$HPSV3_CHECKPOINT_PATH")"
export HPSV3_BASE_MODEL_PATH="$(realpath -e "$HPSV3_BASE_MODEL_PATH")"
export HPSV3_CONFIG_PATH="$(realpath -e "${HPSV3_CONFIG_PATH:-$ROOT_DIR/flow_grpo/HPSv3/hpsv3/config/HPSv3_7B.yaml}")"
[ -f "$HPSV3_CHECKPOINT_PATH" ] && [ -d "$HPSV3_BASE_MODEL_PATH" ] && [ -f "$HPSV3_CONFIG_PATH" ] || {
    printf 'HPSv3 checkpoint/config must be files and the base model must be a directory.\n' >&2
    exit 1
}

export PYTHONPATH="$ROOT_DIR/flow_grpo/HPSv3:$ROOT_DIR/flow_grpo/reward-server${PYTHONPATH:+:$PYTHONPATH}"
export RAY_NODE_IP="${RAY_NODE_IP:-127.0.0.1}"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_ADDRESS="$RAY_NODE_IP:$RAY_PORT"
export REWARD_HTTP_HOST="${REWARD_HTTP_HOST:-127.0.0.1}"
export REWARD_HTTP_PORT="${REWARD_HTTP_PORT:-8080}"
export REWARD_INGRESS_REPLICAS="${REWARD_INGRESS_REPLICAS:-8}"
export RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-60}"
export RAY_ROTATION_MAX_BYTES=104857600
export RAY_ROTATION_BACKUP_COUNT=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

python - <<'PY'
import os
import socket
for name in ("RAY_PORT", "REWARD_HTTP_PORT", "REWARD_INGRESS_REPLICAS", "RAY_START_TIMEOUT"):
    value = int(os.environ[name])
    if value < 1 or (name.endswith("PORT") and value > 65535):
        raise ValueError(f"Invalid {name}")
if os.environ["RAY_PORT"] == os.environ["REWARD_HTTP_PORT"]:
    raise ValueError("Ray and HTTP ports must differ.")
for host, port in ((os.environ["RAY_NODE_IP"], os.environ["RAY_PORT"]),
                   (os.environ["REWARD_HTTP_HOST"], os.environ["REWARD_HTTP_PORT"])):
    with socket.socket() as probe:
        probe.bind((host, int(port)))
PY

RAY_PID=""
SERVICE_PID=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ -n "$SERVICE_PID" ]; then
        kill "$SERVICE_PID" 2>/dev/null || true
        wait "$SERVICE_PID" 2>/dev/null || true
    fi
    if [ -n "$RAY_PID" ]; then
        kill "$RAY_PID" 2>/dev/null || true
        wait "$RAY_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting dedicated Ray Head at %s; HTTP bind %s:%s\n' "$RAY_ADDRESS" "$REWARD_HTTP_HOST" "$REWARD_HTTP_PORT"
ray start --head --node-ip-address="$RAY_NODE_IP" --port="$RAY_PORT" --include-dashboard=false --block &
RAY_PID=$!
deadline=$((SECONDS + RAY_START_TIMEOUT))
while true; do
    if ! kill -0 "$RAY_PID" 2>/dev/null; then
        printf 'Ray Head exited before becoming ready.\n' >&2
        exit 1
    fi
    if timeout 2 ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
        break
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        printf 'Timed out waiting for Ray Head.\n' >&2
        exit 1
    fi
    sleep 1
done
kill -0 "$RAY_PID" 2>/dev/null || { printf 'Ray Head exited.\n' >&2; exit 1; }
python -u "$ROOT_DIR/flow_grpo/reward-server/server_hpsv3.py" &
SERVICE_PID=$!
wait "$SERVICE_PID"
SERVICE_PID=""
