module load conda
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
export HF_HOME="$HF_CACHE_DIR"
conda activate ./myenvs/vllm_env_conda
