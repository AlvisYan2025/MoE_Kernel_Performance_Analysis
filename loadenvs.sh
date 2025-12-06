echo "loading packages.."
module load PrgEnv-gnu
module load python
module load cudatoolkit/12.2
echo "done"
echo "loading venv.." 
source myenvs/vllm_env/bin/activate
echo "done" 