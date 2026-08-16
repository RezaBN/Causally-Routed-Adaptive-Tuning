#!/bin/bash
#SBATCH --account=def-koohi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# =============================================================================
# CRAT ablation runner
# - pair-specific shared Stanza/SIPP caches
# - variant-specific output subdirectories and filenames
# - pair+variant+seed Slurm logs inside each run output directory
# - fixed discovery seed reused across all methods/training seeds for a pair
#
# IMPORTANT: launch with `bash`, NOT directly with `sbatch`.
# The script self-submits so it can create dynamic log paths.
#
# First run for a NEW pair (computes Stanza + SIPP once):
#   SEED=42 DISCOVERY_SEED=42 EPOCHS=4 WALL_TIME=08:00:00 \
#     bash code/crat_mistral_trillium_ablation.sh en-fa full opus100
#
# Later runs reuse the same pair-specific caches:
#   SEED=64 DISCOVERY_SEED=42 EPOCHS=4 WALL_TIME=01:30:00 \
#     bash code/crat_mistral_trillium_ablation.sh en-fa no_kl opus100
# =============================================================================

PAIR="${1:-en-fr}"
ABLATION="${2:-full}"
DATASET_NAME="${3:-opus100}"

SEED="${SEED:-42}"
DISCOVERY_SEED="${DISCOVERY_SEED:-42}"
RUN_MODE="${RUN_MODE:-full}"
WALL_TIME="${WALL_TIME:-08:00:00}"

MORPH_BACKEND="${MORPH_BACKEND:-stanza}"
DETECTION_MODE="${DETECTION_MODE:-sipp}"
HEADS_PER_DIRECTION="${HEADS_PER_DIRECTION:-64}"
PROBE_EXAMPLES="${PROBE_EXAMPLES:-32}"
MIN_PROBE_EXAMPLES="${MIN_PROBE_EXAMPLES:-8}"
CAUSAL_SCORE_EXAMPLES="${CAUSAL_SCORE_EXAMPLES:-16}"
COMPONENT_BATCH_SIZE="${COMPONENT_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
BENCHMARK_EXAMPLES="${BENCHMARK_EXAMPLES:-64}"

LAMBDA_KL="${LAMBDA_KL:-0.10}"
LAMBDA_CAUSAL="${LAMBDA_CAUSAL:-0.10}"
LAMBDA_MORPH="${LAMBDA_MORPH:-0.10}"
LAMBDA_SPARSE="${LAMBDA_SPARSE:-0.002}"

METHODS="crat"                 # internal Python implementation method
VARIANT_NAME=""                # externally meaningful experimental method/variant
DISABLE_INTEGRATION=0
EXTRA_ARGS=()

case "$ABLATION" in
  full)
    VARIANT_NAME="full_crat"
    ;;
  no_kl)
    VARIANT_NAME="no_kl"
    LAMBDA_KL=0
    ;;
  no_causal)
    VARIANT_NAME="no_causal"
    LAMBDA_CAUSAL=0
    ;;
  no_morph)
    VARIANT_NAME="no_morph"
    LAMBDA_MORPH=0
    ;;
  no_sparse)
    VARIANT_NAME="no_sparse"
    LAMBDA_SPARSE=0
    ;;
  no_integration)
    VARIANT_NAME="no_integration"
    DISABLE_INTEGRATION=1
    ;;
  static_role)
    VARIANT_NAME="static_role_sft"
    METHODS="static_role_sft"
    ;;

  # Combined ablations requested for the extended study.
  no_sparse_morph_integration)
    VARIANT_NAME="no_sparse_morph_integration"
    LAMBDA_SPARSE=0
    LAMBDA_MORPH=0
    DISABLE_INTEGRATION=1
    ;;
  no_sparse_morph)
    VARIANT_NAME="no_sparse_morph"
    LAMBDA_SPARSE=0
    LAMBDA_MORPH=0
    ;;
  no_sparse_integration)
    VARIANT_NAME="no_sparse_integration"
    LAMBDA_SPARSE=0
    DISABLE_INTEGRATION=1
    ;;
  no_morph_integration)
    VARIANT_NAME="no_morph_integration"
    LAMBDA_MORPH=0
    DISABLE_INTEGRATION=1
    ;;
  *)
    echo "ERROR: unknown ABLATION='$ABLATION'."
    echo "Allowed: full, no_kl, no_causal, no_morph, no_sparse, no_integration, static_role,"
    echo "         no_sparse_morph_integration, no_sparse_morph, no_sparse_integration, no_morph_integration"
    exit 2
    ;;
esac

if (( DISABLE_INTEGRATION == 1 )); then
  EXTRA_ARGS+=(--disable_integration_adapters)
fi

if [[ "$RUN_MODE" != "full" ]]; then
  echo "ERROR: Ablation experiments must use RUN_MODE=full."
  exit 2
fi
if [[ ! "$PAIR" =~ ^[a-z]{2,3}-[a-z]{2,3}$ ]]; then
  echo "ERROR: invalid language pair '$PAIR'. Expected e.g. en-fr or en-fa."
  exit 2
fi
for item in "SEED:$SEED" "DISCOVERY_SEED:$DISCOVERY_SEED"; do
  name="${item%%:*}"
  value="${item#*:}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value > 4294967295 )); then
    echo "ERROR: $name must be an integer in [0, 4294967295]; got '$value'."
    exit 2
  fi
done

SRC_LANG="${PAIR%%-*}"
TGT_LANG="${PAIR#*-}"
PAIR_TAG="${SRC_LANG}${TGT_LANG}"

ROOT="${CRAT_ROOT:-$SCRATCH/crat}"
ABLATION_ROOT="$ROOT/outputs/ablations"
OUTPUT="$ABLATION_ROOT/mistral7b_${DATASET_NAME}_${PAIR_TAG}_seed${SEED}_${ABLATION}"
LOG_DIR="$OUTPUT/logs"

# Shared caches are isolated by dataset AND exact language pair.
# Example:
#   outputs/ablations/_shared/opus100/en-fa/en-fa_morph_cache_stanza.pkl
#   outputs/ablations/_shared/opus100/en-fa/en-fa_component_discovery.json
PAIR_SHARED_DIR="$ABLATION_ROOT/_shared/$DATASET_NAME/$PAIR"
MORPH_BASENAME="${PAIR}_morph_cache_${MORPH_BACKEND}.pkl"
DISCOVERY_BASENAME="${PAIR}_component_discovery.json"
SHARED_MORPH="$PAIR_SHARED_DIR/$MORPH_BASENAME"
SHARED_DISCOVERY="$PAIR_SHARED_DIR/$DISCOVERY_BASENAME"
CURRENT_MORPH="$OUTPUT/$MORPH_BASENAME"
CURRENT_DISCOVERY="$OUTPUT/$DISCOVERY_BASENAME"

# Every run-specific filename uses the actual experimental variant, not the
# internal implementation label "crat".
ARTIFACT_PREFIX="${PAIR}_seed${SEED}_${VARIANT_NAME}"
LOG_PREFIX="${PAIR}_${VARIANT_NAME}_seed${SEED}"
JOB_NAME="${PAIR_TAG}_s${SEED}_${VARIANT_NAME}"

IS_CANONICAL=0
if [[ "$SEED" == "$DISCOVERY_SEED" && "$ABLATION" == "full" ]]; then
  IS_CANONICAL=1
fi

# ----------------------------------------------------------------------------
# Launcher mode
# ----------------------------------------------------------------------------
if [[ -z "${CRAT_WORKER:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: launch this script with bash rather than sbatch."
    echo "Example: SEED=$SEED EPOCHS=$EPOCHS WALL_TIME=$WALL_TIME bash $0 $PAIR $ABLATION $DATASET_NAME"
    exit 2
  fi

  mkdir -p "$LOG_DIR" "$PAIR_SHARED_DIR"

  # Prevent accidental SIPP/Stanza recomputation by non-canonical jobs.
  if (( IS_CANONICAL == 0 )); then
    if [[ ! -s "$SHARED_MORPH" || ! -s "$SHARED_DISCOVERY" ]]; then
      echo "ERROR: pair-specific shared caches are not ready for $PAIR."
      echo "Expected:"
      echo "  $SHARED_MORPH"
      echo "  $SHARED_DISCOVERY"
      echo
      echo "Run the canonical first job and let it finish before other ablations:"
      echo "  SEED=$DISCOVERY_SEED DISCOVERY_SEED=$DISCOVERY_SEED EPOCHS=$EPOCHS WALL_TIME=08:00:00"
      echo "    bash $0 $PAIR full $DATASET_NAME"
      exit 4
    fi
  fi

  SCRIPT_PATH="$(readlink -f "$0")"
  JOB_ID=$(sbatch --parsable \
    --time="$WALL_TIME" \
    --job-name="$JOB_NAME" \
    --output="$LOG_DIR/${LOG_PREFIX}-%j.out" \
    --error="$LOG_DIR/${LOG_PREFIX}-%j.err" \
    --export=ALL,CRAT_WORKER=1 \
    "$SCRIPT_PATH" "$PAIR" "$ABLATION" "$DATASET_NAME")

  echo "Submitted job: $JOB_ID"
  echo "Pair:          $PAIR"
  echo "Training seed: $SEED"
  echo "Discovery seed:$DISCOVERY_SEED"
  echo "Ablation:      $ABLATION"
  echo "Variant:       $VARIANT_NAME"
  echo "Internal:      $METHODS"
  echo "Output:        $OUTPUT"
  echo "Logs:          $LOG_DIR/${LOG_PREFIX}-${JOB_ID}.{out,err}"
  exit 0
fi

# ----------------------------------------------------------------------------
# Worker mode
# ----------------------------------------------------------------------------
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load arrow/25.0.0
module load cuda/12.6
module load python/3.11.5

CODE="$ROOT/code/crat_mistral_paper_faithful_h100_trillium_ablation.py"
VENV="$ROOT/envs/env_crat"
MODEL="$ROOT/models/Mistral-7B-v0.1"
DATA_DIR="$ROOT/data/$DATASET_NAME/$PAIR"
PROBE_DIR="$ROOT/data/probes"
PROBE_FILE="${PROBE_FILE:-$PROBE_DIR/${PAIR}_bidirectional.jsonl}"
CACHE="$ROOT/cache"
STANZA_DIR="$ROOT/stanza_resources"

cd "$ROOT"
source "$VENV/bin/activate"

export HF_HOME="$CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$CACHE/datasets"
export XDG_CACHE_HOME="$CACHE/xdg"
export STANZA_RESOURCES_DIR="$STANZA_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export TMPDIR="$SLURM_TMPDIR"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/pycache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
  "$OUTPUT" "$LOG_DIR" "$PAIR_SHARED_DIR" "$PROBE_DIR" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
  "$SLURM_TMPDIR/pycache" "$SLURM_TMPDIR/triton" "$SLURM_TMPDIR/torchinductor"

# Build the exact pair-specific bidirectional probe file when absent.
if [[ ! -s "$PROBE_FILE" ]]; then
  FORWARD_PROBE="$PROBE_DIR/${SRC_LANG}-${TGT_LANG}.jsonl"
  REVERSE_PROBE="$PROBE_DIR/${TGT_LANG}-${SRC_LANG}.jsonl"
  if [[ ! -s "$FORWARD_PROBE" || ! -s "$REVERSE_PROBE" ]]; then
    echo "ERROR: pair-specific probe files are missing:"
    echo "  $FORWARD_PROBE"
    echo "  $REVERSE_PROBE"
    exit 3
  fi
  cat "$FORWARD_PROBE" "$REVERSE_PROBE" > "$PROBE_FILE"
fi

validate_discovery_file() {
  local file="$1"
  python - "$file" "$SRC_LANG" "$TGT_LANG" "$DISCOVERY_SEED" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
src, tgt, discovery_seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
d = json.loads(p.read_text(encoding="utf-8"))
sig = d.get("signature", {})
found = (str(sig.get("source_lang", "")).lower(), str(sig.get("target_lang", "")).lower())
expected = (src.lower(), tgt.lower())
if found != expected:
    raise SystemExit(f"PAIR-SAFETY ERROR: discovery {p} is {found[0]}-{found[1]}, expected {src}-{tgt}")
if int(sig.get("seed", -1)) != discovery_seed:
    raise SystemExit(f"DISCOVERY-SEED ERROR: discovery {p} has seed={sig.get('seed')}, expected {discovery_seed}")
print(f"Discovery pair check OK: {src}-{tgt}; discovery_seed={discovery_seed}; file={p}")
PY
}

validate_morph_file() {
  local file="$1"
  python - "$file" "$SRC_LANG" "$TGT_LANG" <<'PY'
import pickle, sys
from pathlib import Path
p = Path(sys.argv[1])
expected = {sys.argv[2].lower(), sys.argv[3].lower()}
with p.open("rb") as f:
    cache = pickle.load(f)
langs = {str(k).split(":", 1)[0].lower() for k in cache.keys()}
if not langs:
    raise SystemExit(f"PAIR-SAFETY ERROR: morphology cache is empty: {p}")
extra = langs - expected
if extra:
    raise SystemExit(f"PAIR-SAFETY ERROR: morphology cache {p} contains foreign languages {sorted(extra)}; expected only {sorted(expected)}")
print(f"Morphology pair check OK: languages={sorted(langs)}; file={p}")
PY
}

# Canonical first run creates missing shared artifacts. If they already exist,
# even the canonical run reuses them instead of recomputing.
REQUIRE_CACHED_DISCOVERY=0
if [[ -s "$SHARED_MORPH" ]]; then
  validate_morph_file "$SHARED_MORPH"
  echo "Reusing SAME-PAIR shared morphology cache: $SHARED_MORPH"
  cp -fp "$SHARED_MORPH" "$CURRENT_MORPH"
fi
if [[ -s "$SHARED_DISCOVERY" ]]; then
  validate_discovery_file "$SHARED_DISCOVERY"
  echo "Reusing SAME-PAIR shared fixed SIPP discovery: $SHARED_DISCOVERY"
  cp -fp "$SHARED_DISCOVERY" "$CURRENT_DISCOVERY"
  REQUIRE_CACHED_DISCOVERY=1
fi

if (( IS_CANONICAL == 0 )); then
  # Launcher already checked presence; repeat in worker for safety.
  test -s "$SHARED_MORPH"
  test -s "$SHARED_DISCOVERY"
  validate_morph_file "$SHARED_MORPH"
  validate_discovery_file "$SHARED_DISCOVERY"
  cp -fp "$SHARED_MORPH" "$CURRENT_MORPH"
  cp -fp "$SHARED_DISCOVERY" "$CURRENT_DISCOVERY"
  REQUIRE_CACHED_DISCOVERY=1
else
  if [[ ! -s "$SHARED_MORPH" ]]; then
    echo "No shared $PAIR morphology cache exists: this canonical run will compute it."
  fi
  if [[ ! -s "$SHARED_DISCOVERY" ]]; then
    echo "No shared $PAIR component discovery exists: this canonical run will run SIPP once."
  fi
fi

# Copy pair-specific shared probe examples for provenance when available.
for f in "$PAIR_SHARED_DIR"/${PAIR}_probe_examples_*.json; do
  [[ -e "$f" ]] || continue
  cp -fp "$f" "$OUTPUT/"
done

echo "========================================================================"
echo "Job ID:                 $SLURM_JOB_ID"
echo "Node:                   $(hostname)"
echo "Started:                $(date)"
echo "Experiment:             CRAT ablation"
echo "Pair:                   $PAIR"
echo "Dataset:                $DATASET_NAME"
echo "Training seed:          $SEED"
echo "Discovery seed:         $DISCOVERY_SEED"
echo "Ablation slug:          $ABLATION"
echo "Variant/method name:    $VARIANT_NAME"
echo "Internal implementation:$METHODS"
echo "Artifact prefix:        $ARTIFACT_PREFIX"
echo "Detection:              $DETECTION_MODE"
echo "Morphology:             $MORPH_BACKEND"
echo "Current morph cache:    $CURRENT_MORPH"
echo "Current discovery:      $CURRENT_DISCOVERY"
echo "Shared pair cache dir:  $PAIR_SHARED_DIR"
echo "Reuse-only discovery:   $REQUIRE_CACHED_DISCOVERY"
echo "Heads/direction:        $HEADS_PER_DIRECTION"
echo "Probes/direction:       $PROBE_EXAMPLES"
echo "Causal examples:        $CAUSAL_SCORE_EXAMPLES"
echo "Epochs:                 $EPOCHS"
echo "Maximum updates:        $MAX_STEPS"
echo "lambda_KL:              $LAMBDA_KL"
echo "lambda_causal:          $LAMBDA_CAUSAL"
echo "lambda_morph:           $LAMBDA_MORPH"
echo "lambda_sparse:          $LAMBDA_SPARSE"
echo "Integration adapters:   $([[ $DISABLE_INTEGRATION -eq 1 ]] && echo disabled || echo enabled)"
echo "Evaluation/direction:   $EVAL_LIMIT"
echo "Model:                  $MODEL"
echo "Output:                 $OUTPUT"
echo "Logs:                   $LOG_DIR"
echo "========================================================================"

python --version
nvidia-smi

test -f "$CODE"
test -x "$VENV/bin/python"
test -f "$MODEL/config.json"
find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .
test -s "$DATA_DIR/train.jsonl"
test -s "$DATA_DIR/validation.jsonl"
test -s "$DATA_DIR/test.jsonl"
test -s "$PROBE_FILE"
python -m py_compile "$CODE"

# Input pair-safety and offline Stanza preflight.
python - <<PY
import json
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

model_path = Path(r"$MODEL")
files = {
    "train": Path(r"$DATA_DIR/train.jsonl"),
    "validation": Path(r"$DATA_DIR/validation.jsonl"),
    "test": Path(r"$DATA_DIR/test.jsonl"),
    "probe": Path(r"$PROBE_FILE"),
}
expected = {r"$SRC_LANG", r"$TGT_LANG"}

config = AutoConfig.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated GPU job.")
print("GPU:", torch.cuda.get_device_name(0))
print("Model layers:", config.num_hidden_layers)
print("Query heads:", config.num_attention_heads)
print("KV heads:", config.num_key_value_heads)
print("Tokenizer size:", len(tokenizer))

for name, path in files.items():
    with path.open(encoding="utf-8") as handle:
        first = json.loads(next(line for line in handle if line.strip()))
    required = {"src", "tgt", "src_lang", "tgt_lang"}
    missing = required - set(first)
    if missing:
        raise RuntimeError(f"{name} is missing fields {sorted(missing)}: {path}")
    observed = {str(first["src_lang"]).lower(), str(first["tgt_lang"]).lower()}
    if not observed.issubset(expected):
        raise RuntimeError(f"PAIR-SAFETY ERROR: {name} contains {observed}; expected only {expected}: {path}")
    print(f"{name}: {path} | {first['src_lang']}-{first['tgt_lang']} | {first['src'][:40]!r} -> {first['tgt'][:40]!r}")

if r"$MORPH_BACKEND" == "stanza":
    import stanza
    for language in [r"$SRC_LANG", r"$TGT_LANG"]:
        text = "این یک آزمایش آفلاین است." if language == "fa" else "Offline Stanza test."
        pipeline = stanza.Pipeline(
            lang=language,
            processors="tokenize,pos",
            model_dir=r"$STANZA_DIR",
            use_gpu=False,
            download_method=None,
            verbose=False,
        )
        doc = pipeline(text)
        print(f"Stanza offline OK: {language}; sentences={len(doc.sentences)}")
PY

PY_ARGS=(
  --model_name "$MODEL"
  --dataset_mode local
  --train_file "$DATA_DIR/train.jsonl"
  --validation_file "$DATA_DIR/validation.jsonl"
  --test_file "$DATA_DIR/test.jsonl"
  --probe_file "$PROBE_FILE"
  --source_lang "$SRC_LANG"
  --target_lang "$TGT_LANG"
  --output_dir "$OUTPUT"
  --artifact_prefix "$ARTIFACT_PREFIX"
  --variant_name "$VARIANT_NAME"
  --morph_cache_file "$CURRENT_MORPH"
  --component_discovery_file "$CURRENT_DISCOVERY"
  --no_auto_download_muse
  --no_require_probe_correct
  --detection_mode "$DETECTION_MODE"
  --discovery_seed "$DISCOVERY_SEED"
  --heads_per_direction "$HEADS_PER_DIRECTION"
  --probe_examples_per_direction "$PROBE_EXAMPLES"
  --min_probe_examples_per_direction "$MIN_PROBE_EXAMPLES"
  --causal_score_examples "$CAUSAL_SCORE_EXAMPLES"
  --component_batch_size "$COMPONENT_BATCH_SIZE"
  --epochs "$EPOCHS"
  --max_steps "$MAX_STEPS"
  --train_batch_size 4
  --gradient_accumulation_steps 32
  --eval_batch_size 8
  --learning_rate 2e-5
  --warmup_ratio 0.02
  --weight_decay 0.1
  --methods "$METHODS"
  --morph_backend "$MORPH_BACKEND"
  --lambda_kl "$LAMBDA_KL"
  --lambda_causal "$LAMBDA_CAUSAL"
  --lambda_morph "$LAMBDA_MORPH"
  --lambda_sparse "$LAMBDA_SPARSE"
  --eval_limit_per_direction "$EVAL_LIMIT"
  --benchmark_examples "$BENCHMARK_EXAMPLES"
  --attn_implementation sdpa
  --detection_attn_implementation eager
  --seed "$SEED"
)
if (( REQUIRE_CACHED_DISCOVERY == 1 )); then
  PY_ARGS+=(--require_cached_discovery)
fi
PY_ARGS+=("${EXTRA_ARGS[@]}")

srun --ntasks=1 python -u "$CODE" "${PY_ARGS[@]}"

# Publish canonical pair caches exactly once after a successful first run.
if (( IS_CANONICAL == 1 )); then
  test -s "$CURRENT_MORPH"
  test -s "$CURRENT_DISCOVERY"
  validate_morph_file "$CURRENT_MORPH"
  validate_discovery_file "$CURRENT_DISCOVERY"

  if [[ ! -s "$SHARED_MORPH" ]]; then
    tmp="$PAIR_SHARED_DIR/.${MORPH_BASENAME}.${SLURM_JOB_ID}.tmp"
    cp -fp "$CURRENT_MORPH" "$tmp"
    mv -f "$tmp" "$SHARED_MORPH"
    echo "Published pair-specific shared morphology cache: $SHARED_MORPH"
  fi
  if [[ ! -s "$SHARED_DISCOVERY" ]]; then
    tmp="$PAIR_SHARED_DIR/.${DISCOVERY_BASENAME}.${SLURM_JOB_ID}.tmp"
    cp -fp "$CURRENT_DISCOVERY" "$tmp"
    mv -f "$tmp" "$SHARED_DISCOVERY"
    echo "Published pair-specific shared component discovery: $SHARED_DISCOVERY"
  fi
  for f in "$OUTPUT"/${PAIR}_probe_examples_*.json; do
    [[ -e "$f" ]] || continue
    cp -fp "$f" "$PAIR_SHARED_DIR/"
  done
fi

RESULT_FILE="$OUTPUT/${ARTIFACT_PREFIX}_comparison_results.csv"
SAMPLE_FILE="$OUTPUT/${ARTIFACT_PREFIX}_translation_samples.csv"

echo "========================================================================"
echo "Finished: $(date)"
echo "Pair:      $PAIR"
echo "Seed:      $SEED"
echo "Ablation:  $ABLATION"
echo "Variant:   $VARIANT_NAME"
echo "Internal:  $METHODS"
echo "Results:   $RESULT_FILE"
echo "Samples:   $SAMPLE_FILE"
echo "Method dir:$OUTPUT/$VARIANT_NAME"
echo "Logs:      $LOG_DIR"
echo "Shared morphology: $SHARED_MORPH"
echo "Shared discovery:  $SHARED_DISCOVERY"
echo "========================================================================"
