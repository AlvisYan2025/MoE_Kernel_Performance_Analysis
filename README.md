(0) on first start, install vllm + other dependencies + model (only need to run once)
first allocate node
```bash 
    salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 --account=m4999
```
then install 
```bash 
    chmod +x install_conda.sh && ./install_conda.sh
```
(1) load environment 
```bash 
    chmod +x loadenvs_conda.sh && source loadenvs_conda.sh
```
(2) run baseline/default moe
先把launch_baseline/launch_defaultAll2All放到 MoE_Kernel_Performance_Analysis/vllm 里面（上面安装的）
然后把这行 HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache" 改成 HF_CACHE_DIR="${SCRIPT_DIR}/../hf_cache" 
```bash
    cd vllm
    chmod +x launch_defaultAll2All.sh && ./launch_defaultAll2All.sh
```


（还没做） -- 
-确认一下上面的安装脚本没问题
-launch_flashinfer.sh  这个还是没跑通，可能是vllm不支持/这个版本的vllm不支持 要继续研究为什么
-benchmark.sh 还没验证能不能用 能用的话就可以做不同batch size/sequence length...的实验了
-load imbalance 实验设计 （先和TA聊 看他怎么说）
-pplx MoE kernel 可以再试试 我觉得这个和flashinfer至少跑通一个吧 不过也许可以留到presentation以后做
- 用benchmark.sh吧不同batch size之类简单的实验跑出来，baseline vs defaultAll2All
- data pipeline + visualization
- slide 
