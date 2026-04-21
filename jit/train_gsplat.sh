#!/bin/bash

source .env

# Usage: ./jit/train_gsplat.sh [JiT-S/8|JiT-B/8|JiT-L/8|JiT-XL/8|...]
#
# Environment variables:
#   Path inputs:
#     OBJ_LIST, GS_DATA_PATH, MEAN_FILE, STD_FILE, CLASS_MAP_PATH,
#     SPHERE2PLANE_PATH, REF_CAMERA_TAR, RESULTS_DIR, RESUME
#   Launch overrides:
#     NUM_GPUS, NUM_MACHINES, MIXED_PRECISION, DYNAMO_BACKEND
#   Hyperparameters:
#     JIT_TRAIN_CONFIG — YAML for train_gsplat.py (default: jit/configs/jit_train_gsplat.yaml)
#     JIT_OVERRIDES_YAML — hot-reload overrides (default: jit/configs/overrides.yaml).
#       Set to empty to disable:  JIT_OVERRIDES_YAML= ./jit/train_gsplat.sh
# Positional $1 is an explicit model override. When unset, the YAML's
# `model:` field is authoritative (via --config). Avoid defaulting MODEL to
# JiT-XL/8 here — passing --model on the CLI would otherwise shadow the YAML.
MODEL_OVERRIDE=${1:-}
JIT_TRAIN_CONFIG=${JIT_TRAIN_CONFIG:-jit/configs/jit_train_gsplat.yaml}
# Unset → default path; explicitly empty → no --overrides_yaml
JIT_OVERRIDES_YAML="${JIT_OVERRIDES_YAML-jit/configs/overrides.yaml}"

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=1
fi
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}

OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-${DIT_GSPLAT_CLASS_MAP:-}}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-${DIT_GSPLAT_REF_CAMERA_TAR:-}}
RESUME=${RESUME:-}

# If RESUME not set via env, check the YAML config for a resume path
if [ -z "$RESUME" ] && [ -f "$JIT_TRAIN_CONFIG" ]; then
    YAML_RESUME=$(python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$JIT_TRAIN_CONFIG')) or {}
v = cfg.get('resume')
if v and str(v).lower() not in ('null', 'none', '~', ''):
    print(v)
" 2>/dev/null)
    if [ -n "$YAML_RESUME" ]; then
        RESUME="$YAML_RESUME"
    fi
fi

for path_var in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE CLASS_MAP_PATH SPHERE2PLANE_PATH REF_CAMERA_TAR; do
    path_value=${!path_var}
    if [ -z "$path_value" ]; then
        echo "Missing required path variable: $path_var" >&2
        exit 1
    fi
    if [ ! -e "$path_value" ]; then
        echo "Configured path does not exist for $path_var: $path_value" >&2
        exit 1
    fi
done

if [ -n "$RESUME" ] && [ ! -e "$RESUME" ]; then
    echo "Configured resume checkpoint does not exist: $RESUME" >&2
    exit 1
fi

# Resolve the effective model for RESULTS_DIR: CLI positional > YAML > fallback.
# (The actual model argument to Python is handled below — this is display only.)
if [ -n "$MODEL_OVERRIDE" ]; then
    EFFECTIVE_MODEL="$MODEL_OVERRIDE"
else
    YAML_MODEL=$(python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$JIT_TRAIN_CONFIG')) or {}
v = cfg.get('model')
if v: print(v)
" 2>/dev/null)
    EFFECTIVE_MODEL="${YAML_MODEL:-JiT-XL/8}"
fi

RESULTS_DIR="output/jit_${EFFECTIVE_MODEL}_results_gsplat"
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_STEM="train_${RUN_TS}_$$"

LOG_DIR="${RESULTS_DIR}"
mkdir -p "$LOG_DIR"
STDOUT_FILE="${LOG_DIR}/${RUN_STEM}.out"
STDERR_FILE="${LOG_DIR}/${RUN_STEM}.err"
PID_FILE="${LOG_DIR}/${RUN_STEM}.pid"

if [ "$NUM_GPUS" -le 1 ]; then
    CMD=(python)
else
    CMD=(
        accelerate launch
        --num_processes "$NUM_GPUS"
        --num_machines "$NUM_MACHINES"
        --multi_gpu
        --mixed_precision "$MIXED_PRECISION"
        --dynamo_backend "$DYNAMO_BACKEND"
    )
fi

PY_ARGS=(jit/train_gsplat.py)
if [ -f "$JIT_TRAIN_CONFIG" ]; then
    PY_ARGS+=(--config "$JIT_TRAIN_CONFIG")
else
    echo "JIT_TRAIN_CONFIG not found: $JIT_TRAIN_CONFIG (set JIT_TRAIN_CONFIG or add the file)" >&2
    exit 1
fi
# Only pass --model when the user explicitly overrode via positional arg.
# Otherwise the YAML's `model:` takes effect.
if [ -n "$MODEL_OVERRIDE" ]; then
    PY_ARGS+=(--model "$MODEL_OVERRIDE")
fi
PY_ARGS+=(
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --class_map "$CLASS_MAP_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --ref_camera_tar "$REF_CAMERA_TAR"
    --mixed_precision "$MIXED_PRECISION"
    --results_dir "$RESULTS_DIR"
)

OVERFIT=${OVERFIT:-0}

if [ -n "$RESUME" ]; then
    PY_ARGS+=(--resume "$RESUME")
fi

if [ "$OVERFIT" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--overfit "$OVERFIT")
fi

if [ -n "$JIT_OVERRIDES_YAML" ]; then
    PY_ARGS+=(--overrides_yaml "$JIT_OVERRIDES_YAML")
fi

echo "Launching in background"
echo "config:  $JIT_TRAIN_CONFIG"
if [ -n "$JIT_OVERRIDES_YAML" ]; then
    echo "overrides: $JIT_OVERRIDES_YAML"
fi
echo "results: $RESULTS_DIR"
echo "stdout: $STDOUT_FILE"
echo "stderr: $STDERR_FILE"
echo "pid:    $PID_FILE"

nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${CMD[@]}" "${PY_ARGS[@]}" >"$STDOUT_FILE" 2>"$STDERR_FILE" < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "Started background job with PID $PID"
