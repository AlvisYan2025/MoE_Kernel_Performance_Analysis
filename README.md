# MoE Kernel Performance Analysis

A comprehensive performance analysis framework for Mixture-of-Experts (MoE) model inference, comparing different kernel implementations and analyzing load balancing effects on performance metrics.

## Overview

This repository contains experiments and tools for benchmarking MoE (Mixture-of-Experts) model inference performance, specifically focusing on:

- **Kernel Comparison**: Baseline (Triton MoE) vs. DefaultAll2All (FlashInfer MoE) implementations
- **Load Imbalance Analysis**: How input data characteristics affect expert routing and performance
- **Performance Metrics**: Time to First Token (TTFT), Time Per Output Token (TPOT), throughput analysis
- **Scalability Studies**: Effects of batch size, sequence length, and parallelization strategies

## Project Structure

```
MoE_Kernel_Performance_Analysis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── scripts/
│   └── load_imbalance_experiment-group12/  # Load imbalance study
│       ├── README.md                 # Detailed experiment documentation
│       ├── benchmark.py              # Benchmarking script
│       ├── generate_datasets.py      # Synthetic dataset generation
│       ├── launch_baseline.sh        # Launch Baseline server
│       ├── launch_defaultAll2All.sh  # Launch DefaultAll2All server
│       ├── client.sh                 # Benchmark client
│       ├── get_real_imbalance.py     # Compute load imbalance (per-step)
│       ├── new_real_imbalance.py     # Compute load imbalance (aggregated)
│       └── datasets/                 # Generated test datasets
│
├── trace_collection/                  # Performance benchmark experiments
│   ├── README.md                     # Experiment overview
│   └── Mixtral8x7B-*/               # Individual experiment directories
│       ├── server.sh                 # Server launch script
│       ├── client.sh                 # Client benchmark script
│       ├── *.yaml                    # Workload configuration
│       ├── results_json/             # Performance metrics
│       └── logs/                     # Server logs and traces
│
├── tools/                             # Metric extraction and visualization
│   ├── main.py                       # Unified metric extraction interface
│   ├── README.md                     # Tools documentation
│   ├── TTFT-group_12/                # Time to First Token tools
│   ├── TPOT-group_12/                # Time Per Output Token tools
│   ├── requestThroughput-group_12/   # Request throughput tools
│   ├── outputThroughput-group_12/    # Output throughput tools
│   └── tokenToExpertAssignment-group_12/  # Expert assignment analysis
│
└── data_analysis_pipeline/            # Dataset analysis and metrics
    ├── dataset_metrics.py            # Imbalance predictor
    ├── generate_datasets.py          # Dataset generation
    └── domain_queries.json           # Domain-specific queries
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for running experiments)
- NERSC Perlmutter access (for trace collection experiments)

### Setup

#### Option 1: Virtual Environment Setup (Recommended)

This method creates a conda virtual environment with vLLM and all dependencies:

1. **Allocate compute node** (NERSC Perlmutter example):
   ```bash
   salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 --account=m4999
   ```

2. **Install virtual environment with vLLM**:
   ```bash
   chmod +x install.sh && ./install.sh
   ```
   
   This script:
   - Creates a conda environment at `./myenvs/vllm_env_conda`
   - Installs Python 3.11 and required packages
   - Clones and installs vLLM (commit 2f13319f)
   - Downloads the Mixtral-8x7B-v0.1 model to `./hf_cache/`
   - Sets up HuggingFace cache directory

3. **Load the environment** (for each new session):
   ```bash
   chmod +x loadenvs.sh && source loadenvs.sh
   ```
   
   This activates the conda environment and sets up the HuggingFace cache directory.

#### Option 2: Manual Setup

1. **Clone the repository** (if applicable)

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - `numpy`, `pandas`, `matplotlib` - Data analysis and visualization
   - `transformers`, `scikit-learn` - Tokenization and NLP analysis
   - `PyYAML` - Configuration file parsing

3. **Install vLLM manually** (if not using virtual environment):
   ```bash
   # Allocate compute node (NERSC Perlmutter example)
   salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 --account=m4999
   
   # Follow vLLM installation instructions
   git clone https://github.com/vllm-project/vllm/
   cd vllm
   pip install -e .
   ```

## Experiments

### 1. Load Imbalance Experiment

**Location**: `scripts/load_imbalance_experiment-group12/`

This experiment studies how input data characteristics affect MoE routing imbalance and correlates this with performance metrics.

**Features**:
- Generates 50 datasets with varying predicted imbalance scores
- Tests both Baseline and DefaultAll2All implementations
- Computes predicted imbalance using `ImbalancePredictor`
- Measures real load imbalance from gate logs
- Analyzes correlation between imbalance and performance

**Quick Start**:
```bash
cd scripts/load_imbalance_experiment-group12

# Generate datasets
python generate_datasets.py

# Start server (choose one)
./launch_baseline.sh
# OR
./launch_defaultAll2All.sh

# Run benchmarks (in another terminal)
./client.sh datasets/001_0.7585.jsonl
```

See `scripts/load_imbalance_experiment-group12/README.md` for detailed documentation.

### 2. Performance Benchmark Experiments

**Location**: `trace_collection/`

Systematic performance benchmarks comparing Baseline vs. DefaultAll2All under different configurations.

**Controlled Variables**:
- **Model Type**: Baseline (TP=4) vs. DefaultAll2All (EP=4)
- **MAX_SEQS**: 8, 16, 32, 64
- **BATCH_TOKENS**: 2048, 4096, 8192, 16384
- **EPLB**: Expert Parallel Load Balancing (on/off for DefaultAll2All)

**Experiment Naming Convention**:
```
Mixtral8x7B-<framework>-Perlmutter[<model>_<max_seqs>_<batch_tokens>]-group12
```

Examples:
- `Mixtral8x7B-vllmTP4-Perlmutter[baseline_32_8192]-group12`
- `Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_64_16384]-group12`

**Running an Experiment**:
```bash
cd trace_collection/Mixtral8x7B-vllmTP4-Perlmutter[baseline_32_8192]-group12

# Start server
./server.sh

# Run client (in another terminal on same node)
./client.sh
```

Results are saved to `results_json/` and logs to `logs/`.

See `trace_collection/README.md` for detailed documentation.

## Tools and Analysis

### Metric Extraction

The `tools/` directory provides scripts to extract and visualize performance metrics.

**Unified Interface**:
```bash
python tools/main.py --trace <trace_directory> --metric <metric_name>
```

Available metrics:
- `TTFT` - Time to First Token
- `TPOT` - Time Per Output Token
- `request_throughput` - Request throughput
- `output_throughput` - Output throughput
- `token_to_expert_assignment` - Expert load distribution

**Examples**:
```bash
# Extract TTFT from an experiment
python tools/main.py --trace trace_collection/Mixtral8x7B-vllmTP4-Perlmutter[baseline_32_8192]-group12 --metric TTFT

# Extract expert assignment metrics
python tools/main.py --trace trace_collection/Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_32_8192]-group12 --metric token_to_expert_assignment
```

### Plotting

Generate comparative plots across experiments:

```bash
# Plot TTFT vs. max-num-seqs and max-num-batched-tokens
python tools/TTFT-group_12/plot.py trace_collection/ graphs/

# Plot TPOT metrics
python tools/TPOT-group_12/plot.py trace_collection/ graphs/

# Plot throughput metrics
python tools/outputThroughput-group_12/plot.py trace_collection/ graphs/
```

Plots show:
- Baseline vs. DefaultAll2All comparisons
- Fixed configuration annotations
- Systematic parameter sweeps

### Imbalance Analysis

The load imbalance experiment includes tools to analyze expert routing:

```bash
cd scripts/load_imbalance_experiment-group12

# Per-step method (older)
python compute_all_real_imbalance.py baseline_gates_logs baseline_real_imbalance.json

# Aggregated method (newer)
python new_compute_all_real_imbalance.py baseline_gates_logs baseline_real_imbalance.json
```

See the experiment README for differences between methods.

## Data Analysis Pipeline

The `data_analysis_pipeline/` directory contains:

- **`dataset_metrics.py`**: `ImbalancePredictor` class that predicts routing imbalance from input characteristics
  - Token-level metrics (repetition, vocabulary diversity, entropy)
  - Semantic similarity metrics
  - N-gram concentration
  - Comprehensive component breakdown

- **`generate_datasets.py`**: Creates synthetic datasets with varying imbalance characteristics
  - Uses domain-specific queries from `domain_queries.json`
  - Multiple generation strategies for wide dispersion
  - Saves datasets with predicted imbalance scores

- **`domain_queries.json`**: 25 domains with ~50 queries each (programming, science, cooking, sports, etc.)

## Performance Metrics

The framework collects and analyzes:

1. **Latency Metrics**:
   - **TTFT** (Time to First Token): Latency until first token generation
   - **TPOT** (Time Per Output Token): Average latency per generated token
   - **ITL** (Inter-token Latency): Latency between consecutive tokens

2. **Throughput Metrics**:
   - **Request Throughput**: Requests processed per second
   - **Input Throughput**: Input tokens processed per second
   - **Output Throughput**: Output tokens generated per second

3. **Load Imbalance Metrics**:
   - **Coefficient of Variation (CV)**: Relative variability of expert loads
   - **Max/Min Ratio**: Imbalance ratio
   - **Entropy**: Distribution uniformity
   - **Token-to-Expert Assignment**: Per-expert load distribution

## Hardware Configuration

Experiments are conducted on:
- **Platform**: NERSC Perlmutter
- **GPUs**: 4x NVIDIA A100
- **Model**: Mixtral-8x7B-v0.1
- **Framework**: vLLM
- **Parallelization**: TP=4 (Baseline) or EP=4 (DefaultAll2All)

## Contributing

When adding new experiments:
1. Follow the directory naming convention in `trace_collection/`
2. Include a `README.md` with experiment details
3. Add workload YAML configuration files
4. Update this README if adding new experiment types

When adding new metrics:
1. Create extraction script in `tools/<metric>-group_12/extract.py`
2. Add validation logic for trace compatibility
3. Update `tools/main.py` with metric mapping
4. Optionally add plotting script

## Documentation

- **Main README**: This file
- **Load Imbalance Experiment**: `scripts/load_imbalance_experiment-group12/README.md`
- **Trace Collection Overview**: `trace_collection/README.md`
- **Tools Documentation**: `tools/README.md`
