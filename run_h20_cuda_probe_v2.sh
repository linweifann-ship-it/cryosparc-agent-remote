#!/bin/bash
set -u

LOG_DIR="/home/lisongyang/cryoagent/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/h20_cuda_probe_${STAMP}.log"

run_section() {
  local title="$1"
  shift
  echo "=== ${title} ==="
  "$@"
  local rc=$?
  echo
  echo "[exit_code] ${rc}"
  echo
}

{
  echo "=== H20 CUDA Probe ==="
  date
  echo
  echo "=== Host ==="
  hostname
  echo

  run_section "Slurm" sinfo
  run_section "H20 Node: nvidia-smi" srun -p g8m768 -w H20a --gres=gpu:1 /usr/bin/nvidia-smi
  run_section "H20 Node: Fabric Manager Process" srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'ps -ef | grep -Ei "fabricmanager|nv-fabric|nvidia-fabric" | grep -v grep || true'
  run_section "H20 Node: Fabric Manager Binary" srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'ls /usr/bin | grep -Ei "fabric" || true'
  run_section "H20 Node: CUDA Libraries" srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'python3 -c "import ctypes; libs=[\"libcuda.so\",\"libcuda.so.1\",\"libcudart.so\",\"libcudart.so.12\"]; [print(lib, \"OK\") if not ctypes.CDLL(lib) else None for lib in libs]"'
  run_section "H20 Node: cryoagent-sft-h20 CUDA Probe" srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'conda run -n cryoagent-sft-h20 python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.version.cuda); print(\"available\", torch.cuda.is_available()); print(\"count\", torch.cuda.device_count()); import sys; \
try: x=torch.tensor([1.0], device=\"cuda\"); print(\"tensor_ok\", x.item()) \
except Exception as e: print(\"tensor_fail\", repr(e))"'
  run_section "H20 Node: cryoagent-sft CUDA Probe" srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'conda run -n cryoagent-sft python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.version.cuda); print(\"available\", torch.cuda.is_available()); print(\"count\", torch.cuda.device_count()); import sys; \
try: x=torch.tensor([1.0], device=\"cuda\"); print(\"tensor_ok\", x.item()) \
except Exception as e: print(\"tensor_fail\", repr(e))"'
} > "${LOG_FILE}" 2>&1

echo "${LOG_FILE}"
