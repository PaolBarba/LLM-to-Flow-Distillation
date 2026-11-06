#!/bin/bash
#SBATCH --job-name=lfd-qwen-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=lfd-qwen-%j.log
#SBATCH --account=IscrC_HRC

# Submit from the repository root; provide your Leonardo account with -A.
set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
REPO_ROOT="$PWD"
APP_PYTHON="${APP_PYTHON:-/leonardo_work/IscrC_HRC/environments/genomics_steering/bin/python}"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/.venv-vllm/bin/vllm}"
VLLM_PYTHON="${VLLM_PYTHON:-$(dirname "$VLLM_BIN")/python}"
VLLM_CUDA_COMPAT_DIR="${VLLM_CUDA_COMPAT_DIR:-$REPO_ROOT/.cache/cuda-compat-13-0/usr/local/cuda-13.0/compat}"
MODEL_PATH="${MODEL_PATH:-/leonardo_scratch/large/userexternal/pbarba00/huggingface/Qwen3-14B}"
FLOWMOL_RUNTIME_HOME="${FLOWMOL_RUNTIME_HOME:-$(dirname "$MODEL_PATH")/FlowMol3-runtime}"
JOB_ROOT="${JOB_ROOT:-$REPO_ROOT/runs/qwen-drd2-${SLURM_JOB_ID}}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MISSING_ENV=0
if [[ ! -x "$APP_PYTHON" ]]; then
  echo "Missing application Python: $APP_PYTHON" >&2
  echo 'Load python/3.11.7, then: python3 -m venv .venv && .venv/bin/python -m pip install -e .' >&2
  MISSING_ENV=1
fi
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "Missing vLLM executable: $VLLM_BIN" >&2
  echo 'Load python/3.11.7, then: python3 -m venv .venv-vllm && .venv-vllm/bin/python -m pip install vllm==0.21.0' >&2
  MISSING_ENV=1
fi
(( MISSING_ENV == 0 )) || exit 1
# An executable Python alone does not mean its dependencies were installed.
"$APP_PYTHON" -c 'import adm, numpy, pydantic, requests' || {
  echo "Application dependencies unavailable; run: $APP_PYTHON -m pip install -e ." >&2
  exit 1
}
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing Qwen model: $MODEL_PATH"; exit 1; }
[[ -f "$FLOWMOL_RUNTIME_HOME/assets/flowmol3/setup_receipt.json" ]] || {
  echo "Missing FlowMol setup receipt: $FLOWMOL_RUNTIME_HOME/assets/flowmol3/setup_receipt.json" >&2
  echo 'Prepare it before submission with: bash jobs/setup_flowmol.sh (using the same MODEL_PATH / FLOWMOL_RUNTIME_HOME overrides).' >&2
  echo 'Or set FLOWMOL_RUNTIME_HOME to an existing prepared runtime.' >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || { echo 'The experiment requires a clean committed checkout. Commit the job files before submitting.'; exit 1; }
[[ ! -e "$JOB_ROOT" ]] || { echo "Choose a fresh JOB_ROOT: $JOB_ROOT"; exit 1; }
mkdir -p "$JOB_ROOT"
export JOB_ROOT

# Preserve Slurm's allocated GPU identifiers, including UUIDs if supplied.
IFS=',' read -r -a JOB_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated GPUs}"
(( ${#JOB_GPUS[@]} >= 2 )) || { echo 'Two visible GPUs are required.'; exit 1; }

# Scope forward-compatibility libraries to the CUDA 13 vLLM environment.
# FlowMol keeps its own CUDA 12 runtime and inherited library search path.
# Include environment tools needed by subprocesses, even without activation.
# Use native sampling to avoid FlashInfer's JIT CUDA compiler requirement.
VLLM_ENV=(env "PATH=$(dirname "$VLLM_BIN"):$PATH" VLLM_USE_FLASHINFER_SAMPLER=0)
if [[ -f "$VLLM_CUDA_COMPAT_DIR/libcuda.so.1" ]]; then
  VLLM_ENV+=("LD_LIBRARY_PATH=$VLLM_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
fi
CUDA_VISIBLE_DEVICES="${JOB_GPUS[0]}" "${VLLM_ENV[@]}" "$VLLM_PYTHON" - <<'PY' || {
import torch
print(f'vLLM PyTorch {torch.__version__}, CUDA {torch.version.cuda}', flush=True)
torch.cuda.init()
print(f'vLLM GPU: {torch.cuda.get_device_name(0)}', flush=True)
PY
  echo "vLLM CUDA initialization failed. Check driver compatibility and VLLM_CUDA_COMPAT_DIR=$VLLM_CUDA_COMPAT_DIR" >&2
  exit 1
}

# The existing provider requires HTTPS. Trust only this job's local certificate.
umask 077
# Avoid inheriting site v3_ca extensions: OpenSSL 1.1.1 can otherwise emit
# duplicate basicConstraints when combined with -addext, breaking verification.
cat > "$JOB_ROOT/tls.cnf" <<'TLS'
[req]
distinguished_name = subject
x509_extensions = server
prompt = no
[subject]
CN = localhost
[server]
subjectAltName = IP:127.0.0.1,DNS:localhost
basicConstraints = critical,CA:TRUE
keyUsage = critical,digitalSignature,keyEncipherment,keyCertSign
extendedKeyUsage = serverAuth
subjectKeyIdentifier = hash
TLS
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
  -config "$JOB_ROOT/tls.cnf" \
  -keyout "$JOB_ROOT/server.key" -out "$JOB_ROOT/server.pem" \
  > "$JOB_ROOT/tls.log" 2>&1
openssl verify -CAfile "$JOB_ROOT/server.pem" "$JOB_ROOT/server.pem"
export REQUESTS_CA_BUNDLE="$JOB_ROOT/server.pem"
export ADM_LLM_PROVIDER=litellm
# This selects the repository's Responses transport; no LiteLLM service is needed.
export LITELLM_API_BASE="https://127.0.0.1:${QWEN_PORT:-8000}/v1"
export LITELLM_API_KEY="$(openssl rand -hex 32)"
export LITELLM_MODEL_MAP='{}'
unset API_KEY_FILE
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"

# Support both older and newer vLLM YaRN configuration syntax.
"${VLLM_ENV[@]}" "$VLLM_BIN" serve --help > "$JOB_ROOT/vllm-help.txt"
if grep -q -- '--rope-scaling' "$JOB_ROOT/vllm-help.txt"; then
  YARN_ARGS=(--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}')
else
  YARN_ARGS=(--hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}')
fi
CUDA_VISIBLE_DEVICES="${JOB_GPUS[0]}" VLLM_API_KEY="$LITELLM_API_KEY" \
  "${VLLM_ENV[@]}" "$VLLM_BIN" serve "$MODEL_PATH" --served-model-name qwen3-14b-local \
  --host 127.0.0.1 --port "${QWEN_PORT:-8000}" \
  --ssl-keyfile "$JOB_ROOT/server.key" --ssl-certfile "$JOB_ROOT/server.pem" \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --enforce-eager --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  "${YARN_ARGS[@]}" > "$JOB_ROOT/vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Fail promptly if serving exits, and bound cold-start waiting to 15 minutes.
READY=0
for ((attempt=0; attempt<180; attempt++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -80 "$JOB_ROOT/vllm.log"; exit 1; }
  if curl --silent --fail --max-time 3 --cacert "$REQUESTS_CA_BUNDLE" \
      "https://127.0.0.1:${QWEN_PORT:-8000}/health" > /dev/null; then
    READY=1
    break
  fi
  sleep 5
done
(( READY == 1 )) || { echo "Qwen startup timed out; see $JOB_ROOT/vllm.log"; exit 1; }

# Materialize a smaller profile outside the tracked source checkout. Hashes and
# the source commit are recorded by the experiment's normal preparation step.
"$APP_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

config = json.loads(Path('domains/molecules/drd2/configs/smoke.json').read_text())
config['experiment_id'] += '-QWEN3-14B'
for role, calls in [('candidate_judge', 128), ('curriculum_controller', 4), ('independent_auditor', 4)]:
    section = config[role]
    section.update(model='qwen3-14b-local', provider='local vLLM Responses',
                   max_output_tokens=8192, maximum_provider_calls=calls)
    section.pop('reasoning_effort', None)
config['candidate_judge']['maximum_coverage_redispatches_per_chunk'] = 2
config['transport'].update(requests_per_minute_per_role=60, maximum_operational_retries=0)
config['geometric_control'].update(maximum_corrections=1, maximum_proposal_attempts=2)
config['recovery'].update(maximum_stage_attempts=1, maximum_job_reentries=1)
config['aggregate_api_budget']['maximum_paid_calls'] = 139
config['registered_compute_budget'].update(maximum_gpu_hours_including_final_gnina=6,
                                           maximum_jobs=1)
Path(os.environ['JOB_ROOT'], 'config.json').write_text(json.dumps(config, indent=2) + '\n')
PY

# Validate structured replies through the real transport before FlowMol work.
CUDA_VISIBLE_DEVICES="${JOB_GPUS[1]}" "$APP_PYTHON" -m adm.preferences.provider_smoke \
  --model qwen3-14b-local --output-dir "$JOB_ROOT/provider-smoke" \
  --calls 1 --maximum-usd 1 --input-usd-per-million 10 \
  --output-usd-per-million 60 --execute

export CUDA_VISIBLE_DEVICES="${JOB_GPUS[1]}"
export MOLECULES_E2E_OUTPUT_ROOT="$JOB_ROOT/experiment"
# This is a hard time cap, not a runtime estimate; timeout returns exit code 124.
timeout --signal=TERM --kill-after=60s 150m \
  "$APP_PYTHON" -m domains.molecules.drd2.run --smoke \
  --config "$JOB_ROOT/config.json" --runtime-home "$FLOWMOL_RUNTIME_HOME" \
  --output-root "$JOB_ROOT/experiment"
