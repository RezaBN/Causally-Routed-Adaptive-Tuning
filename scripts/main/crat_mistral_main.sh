#!/bin/bash
#SBATCH --account=def-koohi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Main CRAT comparison launcher/worker.
#
# Launch with BASH (not sbatch) so the script can build pair/method-specific
# output and log paths before submitting the real SLURM job:
#
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa crat opus100
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa crat_ce opus100
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa crat_ce_kl opus100
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa lora opus100
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa dora opus100
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa full_sft opus100
#   SEED=18 bash code/crat_mistral_trillium_pair.sh en-fa all opus100
#
# Main comparison methods:
#   base, full_sft, targeted_sft, lora, rslora, dora, crat, crat_no_sparse,
#   crat_ce, crat_ce_kl
#
# static_role_sft is intentionally NOT part of this main comparison. It belongs
# to the ablation study.

PAIR="${1:-en-fr}"
METHOD="${2:-crat}"
DATASET_NAME="${3:-opus100}"

RUN_MODE="${RUN_MODE:-full}"
SEED="${SEED:-18}"
DISCOVERY_SEED="${DISCOVERY_SEED:-42}"

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

PEFT_RANK="${PEFT_RANK:-16}"
PEFT_ALPHA="${PEFT_ALPHA:-32}"
PEFT_DROPOUT="${PEFT_DROPOUT:-0.05}"
PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-all-linear}"

FULL_SFT_OPTIMIZER="${FULL_SFT_OPTIMIZER:-adamw8bit}"
FULL_SFT_TRAIN_BATCH="${FULL_SFT_TRAIN_BATCH:-1}"
FULL_SFT_GRAD_ACCUM="${FULL_SFT_GRAD_ACCUM:-128}"
FULL_SFT_EVAL_BATCH="${FULL_SFT_EVAL_BATCH:-4}"

if [[ "$RUN_MODE" == "smoke" ]]; then
    MORPH_BACKEND="simple"
    DETECTION_MODE="activation_norm"
    HEADS_PER_DIRECTION=8
    PROBE_EXAMPLES=8
    MIN_PROBE_EXAMPLES=4
    CAUSAL_SCORE_EXAMPLES=2
    COMPONENT_BATCH_SIZE=8
    EPOCHS=1
    MAX_STEPS=5
    EVAL_LIMIT=40
    BENCHMARK_EXAMPLES=8
elif [[ "$RUN_MODE" != "full" ]]; then
    echo "ERROR: RUN_MODE must be 'full' or 'smoke'."
    exit 2
fi

if [[ ! "$PAIR" =~ ^[a-z]{2,3}-[a-z]{2,3}$ ]]; then
    echo "ERROR: invalid language pair '$PAIR'. Expected e.g. en-fr."
    exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ && "$DISCOVERY_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED and DISCOVERY_SEED must be non-negative integers."
    exit 2
fi
if (( SEED > 4294967295 || DISCOVERY_SEED > 4294967295 )); then
    echo "ERROR: seeds must be <= 4294967295."
    exit 2
fi

SRC_LANG="${PAIR%%-*}"
TGT_LANG="${PAIR#*-}"
PAIR_TAG="${SRC_LANG}${TGT_LANG}"

ROOT="${CRAT_ROOT:-$SCRATCH/crat}"

# Trillium compute nodes require job-time writes to remain on $SCRATCH.  This
# launcher writes logs/caches/probes/results beneath ROOT, so reject accidental
# HOME/PROJECT roots before either the launcher or worker creates files.
case "$(realpath -m "$ROOT")/" in
  "$(realpath -m "$SCRATCH")/"*) ;;
  *)
    echo "ERROR: CRAT_ROOT must be inside \$SCRATCH on Trillium." >&2
    echo "  CRAT_ROOT=$ROOT" >&2
    echo "  SCRATCH=$SCRATCH" >&2
    exit 2
    ;;
esac

CODE="$ROOT/code/crat_mistral_paper_faithful_h100_trillium.py"
VENV="$ROOT/envs/env_crat"
MODEL="$ROOT/models/Mistral-7B-v0.1"
DATA_DIR="$ROOT/data/$DATASET_NAME/$PAIR"
PROBE_DIR="$ROOT/data/probes"
PROBE_FILE="${PROBE_FILE:-$PROBE_DIR/${PAIR}_bidirectional.jsonl}"

OUTPUT_ROOT="$ROOT/outputs/main_comparison"
if [[ "$RUN_MODE" == "smoke" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT}_smoke"
    SHARED_ROOT="${CRAT_SHARED_ROOT:-$ROOT/outputs/_shared_smoke}"
else
    SHARED_ROOT="${CRAT_SHARED_ROOT:-$ROOT/outputs/_shared}"
fi

PAIR_SHARED_DIR="$SHARED_ROOT/$DATASET_NAME/$PAIR"
LEGACY_ABLATION_SHARED="$ROOT/outputs/ablations/_shared/$DATASET_NAME/$PAIR"

MORPH_BASENAME="${PAIR}_morph_cache_${MORPH_BACKEND}.pkl"
DISCOVERY_BASENAME="${PAIR}_component_discovery.json"
SHARED_MORPH="$PAIR_SHARED_DIR/$MORPH_BASENAME"
SHARED_DISCOVERY="$PAIR_SHARED_DIR/$DISCOVERY_BASENAME"

VALID_METHODS=(
  base full_sft targeted_sft
  lora rslora dora crat crat_no_sparse crat_ce crat_ce_kl
)

is_valid_method() {
  local candidate="$1"
  local item
  for item in "${VALID_METHODS[@]}"; do
    [[ "$candidate" == "$item" ]] && return 0
  done
  return 1
}

method_output_dir() {
  local method="$1"
  echo "$OUTPUT_ROOT/mistral7b_${DATASET_NAME}_${PAIR_TAG}_seed${SEED}_${method}"
}

method_wall_time() {
  local method="$1"
  if [[ -n "${WALL_TIME:-}" ]]; then
    echo "$WALL_TIME"
  elif [[ "$method" == "full_sft" ]]; then
    echo "24:00:00"
  else
    echo "08:00:00"
  fi
}

# -----------------------------------------------------------------------------
# Launcher phase: dynamically create method-specific log paths, then sbatch.
# -----------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$OUTPUT_ROOT" "$PAIR_SHARED_DIR"

  # Treat the old ablation shared folder as a SAME-PAIR source for migration.
  discovery_available=0
  if [[ -s "$SHARED_DISCOVERY" ||
        -s "$LEGACY_ABLATION_SHARED/$DISCOVERY_BASENAME" ]]; then
      discovery_available=1
  fi

  submit_one() {
    local method="$1"
    local dependency="${2:-}"
    local out logdir wall jobname
    out="$(method_output_dir "$method")"
    logdir="$out/logs"
    wall="$(method_wall_time "$method")"
    jobname="${PAIR_TAG}_s${SEED}_${method}"
    mkdir -p "$logdir"

    local -a sbatch_args=(
      --parsable
      --job-name="$jobname"
      --output="$logdir/${PAIR}_${method}_seed${SEED}-%j.out"
      --error="$logdir/${PAIR}_${method}_seed${SEED}-%j.err"
      --time="$wall"
      --export=ALL
    )
    if [[ -n "$dependency" ]]; then
      sbatch_args+=(--dependency="afterok:${dependency}")
    fi

    LAST_JOB_ID="$(sbatch "${sbatch_args[@]}" "$0" "$PAIR" "$method" "$DATASET_NAME")"
    echo "Submitted job:   $LAST_JOB_ID"
    echo "Pair:            $PAIR"
    echo "Method:          $method"
    echo "Training seed:   $SEED"
    echo "Discovery seed:  $DISCOVERY_SEED"
    echo "Output:          $out"
    echo "Logs:            $logdir/${PAIR}_${method}_seed${SEED}-${LAST_JOB_ID}.{out,err}"
    [[ -n "$dependency" ]] && echo "Dependency:      afterok:$dependency"
    echo
  }

  if [[ "$METHOD" == "all" || "$METHOD" == "all_no_full" ]]; then
    methods=(base targeted_sft lora rslora dora crat crat_no_sparse crat_ce crat_ce_kl)
    if [[ "$METHOD" == "all" ]]; then
      methods+=(full_sft)
    fi

    # When pair discovery does not exist yet, CRAT is the bootstrap job because
    # it needs both morphology and SIPP and publishes both pair-specific caches.
    bootstrap_job=""
    if (( discovery_available == 0 )); then
      echo "No fixed discovery exists for $PAIR."
      echo "Submitting CRAT first as the pair-cache bootstrap job."
      submit_one crat
      bootstrap_job="$LAST_JOB_ID"
    fi

    for m in "${methods[@]}"; do
      if [[ "$m" == "crat" && -n "$bootstrap_job" ]]; then
        continue
      fi
      dep=""
      # To keep every main-comparison run organized against the same shared
      # pair artifacts, wait for the bootstrap CRAT when the cache is new.
      if [[ -n "$bootstrap_job" ]]; then
        dep="$bootstrap_job"
      fi
      submit_one "$m" "$dep"
    done
    exit 0
  fi

  if ! is_valid_method "$METHOD"; then
    echo "ERROR: unknown METHOD='$METHOD'."
    echo "Allowed: ${VALID_METHODS[*]}, all, all_no_full"
    exit 2
  fi

  case "$METHOD" in
    targeted_sft|crat_ce|crat_ce_kl)
      if (( discovery_available == 0 )); then
        echo "ERROR: $METHOD requires the fixed $PAIR component discovery."
        echo "Run CRAT once first so all discovery-dependent methods use the same cache:"
        echo "  SEED=$SEED DISCOVERY_SEED=$DISCOVERY_SEED bash $0 $PAIR crat $DATASET_NAME"
        exit 4
      fi
      ;;
  esac

  submit_one "$METHOD"
  exit 0
fi

# -----------------------------------------------------------------------------
# Worker phase
# -----------------------------------------------------------------------------

if ! is_valid_method "$METHOD"; then
  echo "ERROR: worker received invalid METHOD='$METHOD'."
  exit 2
fi

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load arrow/25.0.0
module load cuda/12.6
module load python/3.11.5

CACHE="$ROOT/cache"
STANZA_DIR="$ROOT/stanza_resources"
OUTPUT="$(method_output_dir "$METHOD")"
LOG_DIR="$OUTPUT/logs"
ARTIFACT_PREFIX="${PAIR}_seed${SEED}_${METHOD}"
CURRENT_MORPH="$OUTPUT/$MORPH_BASENAME"
CURRENT_DISCOVERY="$OUTPUT/$DISCOVERY_BASENAME"

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
  "$SLURM_TMPDIR/pycache" "$SLURM_TMPDIR/triton" \
  "$SLURM_TMPDIR/torchinductor"

needs_discovery=0
needs_morph=0
case "$METHOD" in
  crat|crat_no_sparse)
    needs_discovery=1
    needs_morph=1
    ;;
  targeted_sft|crat_ce|crat_ce_kl)
    needs_discovery=1
    ;;
esac

# Pair-safety validation for shared/copy artifacts.
validate_discovery_file() {
  local file="$1"
  python - "$file" "$SRC_LANG" "$TGT_LANG" "$DISCOVERY_SEED" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
src, tgt = sys.argv[2], sys.argv[3]
expected_seed = int(sys.argv[4])
d = json.loads(path.read_text(encoding="utf-8"))
sig = d.get("signature", {})
obs = (str(sig.get("source_lang", "")).lower(), str(sig.get("target_lang", "")).lower())
if obs != (src.lower(), tgt.lower()):
    raise SystemExit(
        f"PAIR-SAFETY ERROR: {path} contains {obs[0]}-{obs[1]}, expected {src}-{tgt}"
    )
if int(sig.get("seed", -1)) != expected_seed:
    raise SystemExit(
        f"DISCOVERY-SEED ERROR: {path} uses seed={sig.get('seed')}, expected {expected_seed}"
    )
print(f"Discovery pair/seed OK: {path}")
PY
}

validate_morph_file() {
  local file="$1"
  python - "$file" "$SRC_LANG" "$TGT_LANG" <<'PY'
import pickle, sys
from pathlib import Path
path = Path(sys.argv[1])
allowed = {sys.argv[2].lower(), sys.argv[3].lower()}
with path.open("rb") as f:
    cache = pickle.load(f)
langs = {str(k).split(":", 1)[0].lower() for k in cache.keys()}
bad = sorted(langs - allowed)
missing = sorted(allowed - langs)
if bad or missing:
    raise SystemExit(
        f"PAIR-SAFETY ERROR: morphology cache {path}; "
        f"unexpected={bad}, missing={missing}, expected={sorted(allowed)}"
    )
print(f"Morphology pair OK: {path}; languages={sorted(langs)}; entries={len(cache)}")
PY
}

# Migrate only explicitly pair-named artifacts from the existing ablation shared
# folder. Generic old names are NEVER imported.
if [[ ! -s "$SHARED_MORPH" &&
      -s "$LEGACY_ABLATION_SHARED/$MORPH_BASENAME" ]]; then
  validate_morph_file "$LEGACY_ABLATION_SHARED/$MORPH_BASENAME"
  cp -fp "$LEGACY_ABLATION_SHARED/$MORPH_BASENAME" "$SHARED_MORPH"
  echo "Imported SAME-PAIR morphology cache from ablation shared folder."
fi
if [[ ! -s "$SHARED_DISCOVERY" &&
      -s "$LEGACY_ABLATION_SHARED/$DISCOVERY_BASENAME" ]]; then
  validate_discovery_file "$LEGACY_ABLATION_SHARED/$DISCOVERY_BASENAME"
  cp -fp "$LEGACY_ABLATION_SHARED/$DISCOVERY_BASENAME" "$SHARED_DISCOVERY"
  echo "Imported SAME-PAIR component discovery from ablation shared folder."
fi

# Reuse pair-specific shared artifacts. Copies in each run make provenance
# visible while the canonical source remains under outputs/_shared/.
if [[ -s "$SHARED_MORPH" ]]; then
  validate_morph_file "$SHARED_MORPH"
  cp -fp "$SHARED_MORPH" "$CURRENT_MORPH"
  echo "Reusing SAME-PAIR shared morphology cache: $SHARED_MORPH"
fi
if [[ -s "$SHARED_DISCOVERY" ]]; then
  validate_discovery_file "$SHARED_DISCOVERY"
  cp -fp "$SHARED_DISCOVERY" "$CURRENT_DISCOVERY"
  echo "Reusing SAME-PAIR shared fixed component discovery: $SHARED_DISCOVERY"
fi

if (( needs_discovery == 1 )) && [[ "$METHOD" != "crat" && "$METHOD" != "crat_no_sparse" ]] && [[ ! -s "$CURRENT_DISCOVERY" ]]; then
  echo "ERROR: $METHOD requires pair-specific discovery, but it is missing."
  echo "Expected: $SHARED_DISCOVERY"
  exit 4
fi

# Build pair-specific bidirectional probes only when discovery may be needed.
if (( needs_discovery == 1 )) && [[ ! -s "$PROBE_FILE" ]]; then
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

echo "========================================================================"
echo "Job ID:                 $SLURM_JOB_ID"
echo "Node:                   $(hostname)"
echo "Started:                $(date)"
echo "Experiment:             Main CRAT comparison"
echo "Pair:                   $PAIR"
echo "Dataset:                $DATASET_NAME"
echo "Method:                 $METHOD"
echo "Training seed:          $SEED"
echo "Discovery seed:         $DISCOVERY_SEED"
echo "Detection:              $DETECTION_MODE"
echo "Morphology:             $MORPH_BACKEND"
echo "Heads/direction:        $HEADS_PER_DIRECTION"
echo "Probes/direction:       $PROBE_EXAMPLES"
echo "Causal examples:        $CAUSAL_SCORE_EXAMPLES"
echo "Epochs:                 $EPOCHS"
echo "Maximum updates:        $MAX_STEPS"
echo "PEFT rank/alpha:        $PEFT_RANK / $PEFT_ALPHA"
echo "PEFT target modules:    $PEFT_TARGET_MODULES"
echo "Output:                 $OUTPUT"
echo "Logs:                   $LOG_DIR"
echo "Shared pair cache dir:  $PAIR_SHARED_DIR"
echo "Shared morphology:      $SHARED_MORPH"
echo "Shared discovery:       $SHARED_DISCOVERY"
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

if (( needs_discovery == 1 )); then
  test -s "$PROBE_FILE"
fi

python -m py_compile "$CODE"

if [[ "$METHOD" == "lora" || "$METHOD" == "rslora" || "$METHOD" == "dora" ]]; then
  python - <<'PY'
import peft
print("PEFT version:", getattr(peft, "__version__", "unknown"))
PY
fi

if [[ "$METHOD" == "full_sft" && "$FULL_SFT_OPTIMIZER" == "adamw8bit" ]]; then
  python - <<'PY'
import bitsandbytes
print("bitsandbytes version:", getattr(bitsandbytes, "__version__", "unknown"))
PY
fi

python - <<PY
import json
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

model_path = Path(r"$MODEL")
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, use_fast=True
)

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated GPU job.")
print("GPU:", torch.cuda.get_device_name(0))
print("Model layers:", config.num_hidden_layers)
print("Query heads:", config.num_attention_heads)
print("KV heads:", config.num_key_value_heads)
print("Tokenizer size:", len(tokenizer))

for name in ("train", "validation", "test"):
    path = Path(r"$DATA_DIR") / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        first = json.loads(next(line for line in handle if line.strip()))
    required = {"src", "tgt", "src_lang", "tgt_lang"}
    missing = required - set(first)
    if missing:
        raise RuntimeError(f"{name} is missing fields {sorted(missing)}: {path}")
    print(
        f"{name}: {path} | {first['src_lang']}-{first['tgt_lang']} | "
        f"{first['src'][:40]!r} -> {first['tgt'][:40]!r}"
    )

if r"$MORPH_BACKEND" == "stanza" and r"$METHOD" in {"crat", "crat_no_sparse"}:
    import stanza
    for language in [r"$SRC_LANG", r"$TGT_LANG"]:
        pipeline = stanza.Pipeline(
            lang=language,
            processors="tokenize,pos",
            model_dir=r"$STANZA_DIR",
            use_gpu=False,
            download_method=None,
            verbose=False,
        )
        doc = pipeline("Offline Stanza test.")
        print(f"Stanza offline OK: {language}; sentences={len(doc.sentences)}")
PY

TRAIN_BATCH=4
GRAD_ACCUM=32
EVAL_BATCH=8
EXTRA_ARGS=()

if [[ "$METHOD" == "full_sft" ]]; then
  TRAIN_BATCH="$FULL_SFT_TRAIN_BATCH"
  GRAD_ACCUM="$FULL_SFT_GRAD_ACCUM"
  EVAL_BATCH="$FULL_SFT_EVAL_BATCH"
  EXTRA_ARGS+=(--gradient_checkpointing --full_sft_optimizer "$FULL_SFT_OPTIMIZER")
fi

if (( needs_morph == 1 )); then
  EXTRA_ARGS+=(--morph_cache_file "$CURRENT_MORPH")
fi

if (( needs_discovery == 1 )); then
  EXTRA_ARGS+=(--probe_file "$PROBE_FILE")
  EXTRA_ARGS+=(--component_discovery_file "$CURRENT_DISCOVERY")
  if [[ -s "$CURRENT_DISCOVERY" ]]; then
    EXTRA_ARGS+=(--require_cached_discovery)
  fi
fi

# crat_ce and crat_ce_kl are implemented directly by the new Python script:
#   crat_ce    -> CE only
#   crat_ce_kl -> CE + KL
# Both reuse the same fixed pair-specific component discovery; neither requires
# the morphology cache.

# Main-comparison CRAT variant: identical CRAT architecture/objectives except
# the routing sparsity loss is disabled.
if [[ "$METHOD" == "crat_no_sparse" ]]; then
  EXTRA_ARGS+=(--lambda_sparse 0.0)
fi

srun --ntasks=1 python -u "$CODE" \
  --model_name "$MODEL" \
  --dataset_mode local \
  --train_file "$DATA_DIR/train.jsonl" \
  --validation_file "$DATA_DIR/validation.jsonl" \
  --test_file "$DATA_DIR/test.jsonl" \
  --source_lang "$SRC_LANG" \
  --target_lang "$TGT_LANG" \
  --output_dir "$OUTPUT" \
  --artifact_prefix "$ARTIFACT_PREFIX" \
  --no_auto_download_muse \
  --no_require_probe_correct \
  --detection_mode "$DETECTION_MODE" \
  --discovery_seed "$DISCOVERY_SEED" \
  --heads_per_direction "$HEADS_PER_DIRECTION" \
  --probe_examples_per_direction "$PROBE_EXAMPLES" \
  --min_probe_examples_per_direction "$MIN_PROBE_EXAMPLES" \
  --causal_score_examples "$CAUSAL_SCORE_EXAMPLES" \
  --component_batch_size "$COMPONENT_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --train_batch_size "$TRAIN_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --eval_batch_size "$EVAL_BATCH" \
  --learning_rate 2e-5 \
  --warmup_ratio 0.02 \
  --weight_decay 0.1 \
  --methods "$METHOD" \
  --morph_backend "$MORPH_BACKEND" \
  --peft_rank "$PEFT_RANK" \
  --peft_alpha "$PEFT_ALPHA" \
  --peft_dropout "$PEFT_DROPOUT" \
  --peft_target_modules "$PEFT_TARGET_MODULES" \
  --eval_limit_per_direction "$EVAL_LIMIT" \
  --benchmark_examples "$BENCHMARK_EXAMPLES" \
  --attn_implementation sdpa \
  --detection_attn_implementation eager \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

# Publish newly computed pair artifacts only after a successful run.
if [[ -s "$CURRENT_MORPH" && ! -s "$SHARED_MORPH" ]]; then
  validate_morph_file "$CURRENT_MORPH"
  tmp="$PAIR_SHARED_DIR/.${MORPH_BASENAME}.${SLURM_JOB_ID}.tmp"
  cp -fp "$CURRENT_MORPH" "$tmp"
  mv -f "$tmp" "$SHARED_MORPH"
  echo "Published pair-specific shared morphology cache: $SHARED_MORPH"
fi

if [[ -s "$CURRENT_DISCOVERY" && ! -s "$SHARED_DISCOVERY" ]]; then
  validate_discovery_file "$CURRENT_DISCOVERY"
  tmp="$PAIR_SHARED_DIR/.${DISCOVERY_BASENAME}.${SLURM_JOB_ID}.tmp"
  cp -fp "$CURRENT_DISCOVERY" "$tmp"
  mv -f "$tmp" "$SHARED_DISCOVERY"
  echo "Published pair-specific shared component discovery: $SHARED_DISCOVERY"
fi

echo "========================================================================"
echo "Finished: $(date)"
echo "Pair:     $PAIR"
echo "Method:   $METHOD"
echo "Seed:     $SEED"
echo "Results:  $OUTPUT/${ARTIFACT_PREFIX}_comparison_results.csv"
echo "========================================================================"
