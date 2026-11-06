# Short local-Qwen experiment on Leonardo

`qwen_drd2_smoke.sbatch` runs only the DRD2 molecular smoke task: 32 initial
candidates, one alignment round, at most one DPO update of 16 training steps,
and 32 samples per evaluation arm. Goal proposals are capped at two; scientific
acceptance gates remain active, so the run may stop before an update if Qwen
does not produce usable preferences. Docking and final-goal continuation are off.
This is an engineering pilot, not a reproduction of the paper's results.

The allocation requests two GPUs for up to three hours on `boost_usr_prod`:
one for the local Qwen3-14B teacher and one for FlowMol3. The experiment itself
has a 150-minute hard timeout. Completion time depends on model loading, GPU
speed, and teacher responses; the timeout can interrupt an unfinished run.

Prepare these dependencies before submission (downloads are not part of the job):

1. Use the existing application environment at
   `/leonardo_work/IscrC_HRC/environments/genomics_steering` (the job defaults
   to its `bin/python`). Set `APP_PYTHON` to use another environment.
2. With [Pixi](https://pixi.prefix.dev/latest/installation/) on PATH, run
   `bash jobs/setup_flowmol.sh` on a node with internet access. It downloads
   FlowMol3 and installs its separate pinned runtime on the same scratch
   filesystem as Qwen, by default at
   `/leonardo_scratch/large/userexternal/pbarba00/huggingface/FlowMol3-runtime`.
   The checkpoint, source, environment, and package cache all stay there.
   The helper also finds Pixi installed in `.cache/pixi/bin`.
3. Install a recent CUDA-compatible vLLM in a **separate** environment, normally
   `.venv-vllm`. It must support `/v1/responses` with JSON schema output and
   `--default-chat-template-kwargs`. Keep its PyTorch dependencies separate from
   the application and FlowMol environments. Set `VLLM_BIN` if installed elsewhere.
4. Have `openssl`, `curl`, and GNU `timeout` available on compute nodes. Load
   any site-required CUDA/Python modules in your submission environment.
5. Commit these files and any other source edits: the experiment enforces a
   clean, committed checkout. Keep environment directories ignored or outside
   the checkout.

On Leonardo, use the existing application environment and install vLLM
separately on a node with internet access before submitting:

```bash
module load python/3.11.7
export APP_PYTHON=/leonardo_work/IscrC_HRC/environments/genomics_steering/bin/python
# If the project dependencies are not already installed:
"$APP_PYTHON" -m pip install -e .
python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install 'vllm==0.21.0'
```

Keep this Python module loaded when submitting so Slurm inherits its library
paths. The environments above do not install the separate FlowMol runtime;
complete step 2 as well. If the log reports a missing executable, check the
printed path or set `APP_PYTHON` / `VLLM_BIN` to an existing installation.

The PyPI vLLM build installs CUDA 13, whereas Leonardo's R535 nodes expose
CUDA driver API 12.2. NVIDIA's RHEL8 `cuda-compat-13-0` libraries are already
installed in `.cache/cuda-compat-13-0/usr/local/cuda-13.0/compat` in this checkout.
The job adds these libraries to `LD_LIBRARY_PATH` only for vLLM
and checks CUDA initialization before starting the server. This follows
[NVIDIA's forward compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
for R535 and data center GPUs. FlowMol retains its separate CUDA 12 environment.
Set `VLLM_CUDA_COMPAT_DIR` to use a different compatibility library directory,
and `VLLM_PYTHON` if the interpreter is not beside `VLLM_BIN`.

Submit **from the repository root** after setup and committing source changes:

```bash
sbatch -A IscrC_HRC jobs/qwen_drd2_smoke.sbatch
```

Optional environment settings are `APP_PYTHON`, `VLLM_BIN`, `MODEL_PATH`,
`FLOWMOL_RUNTIME_HOME`, `JOB_ROOT`, and `QWEN_PORT`. Slurm partition/resource
directives can be overridden on the `sbatch` command line. `JOB_ROOT` must be
new and outside tracked source paths (the default is under ignored `runs/`).
`FLOWMOL_RUNTIME_HOME` defaults to `FlowMol3-runtime` beside `MODEL_PATH`;
use the same overrides for setup and submission. Both models run in the same
`boost_usr_prod` allocation, on separate GPUs.

The model defaults to
`/leonardo_scratch/large/userexternal/pbarba00/huggingface/Qwen3-14B`.
YaRN is enabled at factor 4, original context 32,768, served context 131,072;
the job supports old `--rope-scaling` and new `--hf-overrides` syntax. These
settings follow [Qwen's guidance](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
and [vLLM's context extension documentation](https://docs.vllm.ai/en/v0.21.0/features/context_extension/).
Non-thinking mode and an 8,192-token answer cap reduce generation time.

The repository's `litellm` provider setting selects its existing Responses
transport, pointed directly at local vLLM; no LiteLLM installation, cloud key,
or external LLM is used. A per-job certificate keeps its HTTPS checks enabled.
The inherited dollar estimates are accounting guards, not cloud charges.
The job checks one schema-constrained response before starting FlowMol.

Console output is `lfd-qwen-JOBID.log`. Server logs, the resolved pilot profile,
and the transport check are in `runs/qwen-drd2-JOBID/`; experiment outputs and
LLM traces are under its `experiment/` directory. The Qwen server is stopped
when the script exits. Inspect `vllm.log` first if startup or the transport
check fails. The script has not been validated in a live GPU allocation.

The smoke job uses the PyTorch-native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`)
to avoid FlashInfer sampling-kernel compilation during startup. It also puts
the vLLM environment’s `bin` directory on PATH for serving subprocesses.

If the teacher omits a molecule judgment, the host retains the valid rows and
requests each missing molecule individually. The smoke profile permits at most
two recovery calls per original batch, within the existing API budgets.
Duplicate or unexpected IDs remain validation errors.
