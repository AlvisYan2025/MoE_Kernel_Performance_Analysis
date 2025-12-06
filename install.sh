#salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 --account=m4999
module load PrgEnv-gnu
module load python
module load cudatoolkit/12.2
python3 -m venv myenvs/vllm_env
source myenvs/vllm_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt