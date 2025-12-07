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

1. 启动后端服务器

先进入交互节点并加载环境，然后在不同端口启动不同的后端。

naive baseline 使用 8000 端口：
```bash
./launch_baseline.sh
```
defaultAll2All 使用 8000 端口：
```bash
./launch_defaultAll2All.sh
```

运行后端时请保持该 terminal 不要关闭。

⸻

2. 运行 benchmark（批量测试）

使用 benchmark_batch_sweep_json_singlefile.py 执行多 batch-size 的 sweep。

测试 naive baseline：
```bash
python benchmark_batch_sweep_json_singlefile.py –port 8001 –mode naive –out results_json/naive.json
```

测试 defaultAll2All：
```bash
python benchmark_batch_sweep_json_singlefile.py –port 8000 –mode defaultAll2All –out results_json/default.json
```

脚本会自动测试多个 batch size（默认为 1、4、8、16、32），并将所有结果保存在一个 JSON 文件里。

⸻

3. Benchmark 数据存放位置

所有测试结果会保存在：
```bash
results_json/
```

例如：
```bash
results_json/naive.json
results_json/default.json
```

文件内容包含平均延迟、P50、P95、TTFT、TPOT 等指标。

⸻

4. 绘图（可选）

要根据 JSON 画出 latency、TTFT、TPOT 曲线：

```bash
python plot_results.py
```

生成的图像会放在：
```bash
graphs/
```

⸻

5. 总结
	•	launch_baseline.sh / launch_defaultAll2All.sh：启动不同后端
	•	benchmark_batch_sweep_json_singlefile.py：执行性能测试
	•	results_json/：保存所有测试数据
	•	graphs/：保存所有图表


（还没做） -- 
-确认一下上面的安装脚本没问题
-launch_flashinfer.sh  这个还是没跑通，可能是vllm不支持/这个版本的vllm不支持 要继续研究为什么
-benchmark.sh 还没验证能不能用 能用的话就可以做不同batch size/sequence length...的实验了
-load imbalance 实验设计 （先和TA聊 看他怎么说）
-pplx MoE kernel 可以再试试 我觉得这个和flashinfer至少跑通一个吧 不过也许可以留到presentation以后做
- 用benchmark.sh吧不同batch size之类简单的实验跑出来，baseline vs defaultAll2All
- data pipeline + visualization
- slide 
