#allocate gpu first 
#salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 --account=m4999
module load conda
echo "creating virtual environment.."
conda create --prefix ./myenvs/vllm_env_conda python=3.11 -y
conda activate ./myenvs/vllm_env_conda
python3 -m pip install --upgrade pip
#python3 -m pip install -r requirements.txt
echo "done" 


git clone https://github.com/vllm-project/vllm/
cd vllm
git checkout v0.11.0
python3 -m pip install setuptools-scm setuptools wheel ninja packaging cmake
module load gcc/11.2.0
module load cuda
python3 -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

#git reset --hard 2f13319f47eb9a78b471c5ced0fcf90862cd16a2
#VLLM_USE_PRECOMPILED=1 python3 -m pip install -e .
python3 -m pip install -e . --no-build-isolation
cd .. 

#install model (mistralai/Mixtral-8x7B-v0.1)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
export HF_HOME="$HF_CACHE_DIR"
mkdir -p "$HF_CACHE_DIR"
echo "HuggingFace cache directory: $HF_CACHE_DIR"
MODEL=${1:-"mistralai/Mixtral-8x7B-v0.1"} #change model here 

python3 - << EOF
import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
from huggingface_hub import snapshot_download

model_id = "${MODEL}"
cache_dir = os.environ["HF_HOME"]

print("Cache directory:", cache_dir)

# 1. Download config
print("\n[1/5] Downloading config...")
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print("✓ Config downloaded!")

# 2. Download tokenizer
print("\n[2/5] Downloading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("✓ Tokenizer downloaded!")

# 3. Download generation config (if exists)
print("\n[3/5] Downloading generation config...")
try:
    GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Generation config downloaded!")
except Exception as e:
    print("⚠ No generation config found:", e)

# 4. Snapshot download ALL model files
print("\n[4/5] Downloading model weights via snapshot_download...")
local_dir = snapshot_download(
    repo_id=model_id,
    repo_type="model",
    local_dir=cache_dir,
    local_dir_use_symlinks=False,
)
print("✓ Model weights downloaded!")
print("Saved to:", local_dir)

# 5. Verify downloaded files
print("\n[5/5] Verifying download...")
import glob
all_files = glob.glob(os.path.join(local_dir, "**"), recursive=True)
print(f"✓ Total files downloaded: {len(all_files)}")

print("\n🎉 Model download complete!")
EOF

echo "done"