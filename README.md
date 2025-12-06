(0) on first start, install vllm + other dependencies + model (only need to run once)
first allocate node
```bash 
    salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 --account=m4999
```
then install 
```bash 
    chmod +x install.sh && ./install.sh
```
(1) load environment 
```bash 
    chmod +x loadenvs.sh && source loadenvs.sh
```
(2) run some launcher 
```bash
    chmod +x launch_flashinfer.sh && ./launch_flashinfer.sh
```