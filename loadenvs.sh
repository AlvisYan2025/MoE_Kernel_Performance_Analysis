echo "loading packages.."
module load PrgEnv-gnu
module load python
module load cudatoolkit/12.2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
export HF_HOME="$HF_CACHE_DIR"
echo "done"
echo "loading venv.." 
source myenvs/vllm_env/bin/activate
echo "done" 