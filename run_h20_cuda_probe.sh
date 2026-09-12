#!/bin/bash
set -euo pipefail

LOG_DIR="/home/lisongyang/cryoagent/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/h20_cuda_probe_${STAMP}.log"

{
  echo "=== H20 CUDA Probe ==="
  date
  echo
  echo "=== Host ==="
  hostname
  echo
  echo "=== Slurm ==="
  sinfo
  echo
  echo "=== H20 Node: nvidia-smi ==="
  srun -p g8m768 -w H20a --gres=gpu:1 /usr/bin/nvidia-smi
  echo
  echo "=== H20 Node: Fabric Manager Process ==="
  srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'ps -ef | grep -Ei "fabricmanager|nv-fabric|nvidia-fabric" | grep -v grep || true'
  echo
  echo "=== H20 Node: Fabric Manager Binary ==="
  srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'ls /usr/bin | grep -Ei "fabric" || true'
  echo
  echo "=== H20 Node: CUDA Libraries ==="
  srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'python3 -c "import ctypes\nfor lib in [\"libcuda.so\", \"libcuda.so.1\", \"libcudart.so\", \"libcudart.so.12\"]:\n    try:\n        ctypes.CDLL(lib)\n        print(lib, \"OK\")\n    except Exception as e:\n        print(lib, \"FAIL\", e)"'
  echo
  echo "=== H20 Node: cryoagent-sft-h20 CUDA Probe ==="
  srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'conda run -n cryoagent-sft-h20 python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.version.cuda); print(\"available\", torch.cuda.is_available()); print(\"count\", torch.cuda.device_count());\ntry:\n    x=torch.tensor([1.0], device=\"cuda\")\n    print(\"tensor_ok\", x.item())\nexcept Exception as e:\n    print(\"tensor_fail\", repr(e))"'
  echo
  echo "=== H20 Node: cryoagent-sft CUDA Probe ==="
  srun -p g8m768 -w H20a --gres=gpu:1 bash -lc 'conda run -n cryoagent-sft python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.version.cuda); print(\"available\", torch.cuda.is_available()); print(\"count\", torch.cuda.device_count());\ntry:\n    x=torch.tensor([1.0], device=\"cuda\")\n    print(\"tensor_ok\", x.item())\nexcept Exception as e:\n    print(\"tensor_fail\", repr(e))"'
} > "${LOG_FILE}" 2>&1

echo "${LOG_FILE}"
