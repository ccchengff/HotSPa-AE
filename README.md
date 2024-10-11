# HotSPa (SOSP '24 Artifact Evaluation)

Codes for our submission entitled *Enabling Parallelism Hot Switching for Efficient Training of Large Language Models*. We built a prototype DL system `Hetu` for `HotSPa` and prepared scripts to run the training with parallelism hot switching and static parallelism strategies, respectively.

## Precise Scripts to Reproduce Experiments

#### Technical Requirements

**8~32 x A800-80G GPUs** is required for our paper's experiments.


#### One-click Script to Reproduce Performance and Compare with Baseline

A regular 8xA100-40G machine is provided for small-scale evaluations (unavailable now).

Now we provide a one-clicked script named `exprs_for_A100_40G.sh` to reproduce the performance of parallelism hot switching for `HotSPa`, and also compare with `Megatron-LM` as baseline.

- two one-click scripts for Baseline and HotSPa
    ~~~bash
    root_folder
    ├── Baselines
    │   └── Megatron-LM
    │       ├── benchmark
    │       │   ├── exprs_for_A100_40G.sh # (*) one-click scirpt for baseline
    │       │   └── llama_baseline.sh # main script for baseline
    │       ├── ...
    │       └── ...
    ├── HotSPa
    │   └── hotspa
    │       ├── scripts
    │       │   ├── exprs_for_A100_40G.sh # (*) one-click scirpt for hotspa
    │       │   ├── llama_static_parallel.sh # main scirpt for static
    │       │   └── llama_hot_switch.sh # main script for hot switch
    │       ├── ...
    │       └── ...
    │   └── hetu # graph compiler and hot switch planner
    │   └── ...

    ~~~
- one-click script to **reproduce** the performance of parallelism hot switching for `HotSPa` 

    ~~~bash
    cd root_folder/HotSPa/hotspa
    conda activate hotspa
    bash scripts/exprs_for_A100_40G.sh
    ~~~

- one-click script to compare with `Megatron-LM` as baseline

    ~~~bash
    cd root_folder/Baselines/Megatron-LM
    conda activate megatron
    bash benchmark/exprs_for_A100_40G.sh
    ~~~

#### Screencast for HotSPa reproduce and Baseline Compare

we also **record the screencast** to show the whole process of environment init, `HotSPa reproduce` and `Baseline compare`

- HotSPa reproduce: https://drive.google.com/file/d/1FfXgzdlk8yuawi7CRBrnSi3iTEDZru8Z/view?usp=sharing
- Baseline compare: https://drive.google.com/file/d/19yMvBnDt5LLhg9QXiMAQkaPpGNwoz8fI/view?usp=sharing

#### Reproduce the Exact Experiments from the Paper

now we provide all the scripts to reproduce the exact experiments from the paper

~~~bash
HotSPa/hotspa/scripts
├── exprs_for_A100_40G.sh # (*) one-click scirpt for hotspa evaluation on A100-40G
├── exprs_for_paper # (*) scirpts to reproduce all the exact experiments from the paper on 4x8 A800-80G
│   ├── commoncrawl
│   │   ├── commoncrawl_13b_16gpus.sh
│   │   ├── commoncrawl_13b_8gpus.sh
│   │   ├── commoncrawl_32b_32gpus.sh
│   │   ├── commoncrawl_7b_16gpus.sh
│   │   └── commoncrawl_7b_8gpus.sh
│   ├── github
│   │   ├── github_13b_16gpus.sh
│   │   ├── github_13b_8gpus.sh
│   │   ├── github_32b_32gpus.sh
│   │   ├── github_7b_16gpus.sh
│   │   └── github_7b_8gpus.sh
│   └── llama_hot_switch.sh
├── llama_hot_switch.sh
└── llama_static_parallel.sh
~~~

## Details

### File Tree

~~~bash
    ...
    hetu # the source code implemented by C++
    hotspa # the python script for `HotSPa` e2e training with parallelism hot switching
    ├── data # related to section `Dataset Preparation`
    │   ├── merges.txt
    │   ├── vocab.json
    │   └── web
    │       ├── refinedweb0_cache.pkl
    │       ├── refinedweb0.json
    │       └── refinedweb0.parquet
    ├── data_utils # related to section `Dataset Preparation`
    │   ├── create_web_dataset.py
    │   ├── create_web_dataset.sh
    │   ├── __init__.py
    │   ├── llama_dataloader.py
    │   └── llama_dataset.py
    ├── ds_parallel_config # the dist configs for `HotSPa`, auto-gen
    │   ├── generate_gpt_3d_config.py
    │   ├── generate_gpt_hetero_3d_config.py
    │   ├── generate_llama_3d_config.py
    │   └── gpus8
    │       ├── 7b
    │       │   ├── dp1_tp4_pp2.json
    │       │   ├── dp2_tp2_pp2.json
    │       │   └── dp8_tp1_pp1.json
    │   └── gpus16
    │       │   └── xxx.json
    │   └── gpus32    
    │           └── xxx.json
    │           
    ├── env # related to section `Build & Compile Hetu`
    │   └── envs.sh
    ├── hetu_llama_multi_ds_parallel_symbolic_sp.py
    ├── hostfile
    ├── llama_config.py
    ├── llama_hot_switch.py
    ├── README.md
    ├── scripts
    │   ├── llama_static_parallel.sh # the script for parallelism hot switching
    │   ├── llama_hot_switch.sh # the script for static parallelism
    │   └── exprs_for_A100_40G.sh # the one-click script to reproduce performance
    └── tokenizer
        ├── gpt2_tokenization.py
        ├── __init__.py
        └── tokenizer.py
    build # related to section `Build & Compile Hetu`
~~~

### Build & Compile Hetu

We use `cmake>=3.24` and `gcc-9.5.0` (not support gcc-10) to compile the `Hetu` system. Related third-party packages like `flash-attn`, `onednn`, `cutlass` have been prepared and will be compiled automatically. You may also configure the path to pre-built modules by modifing the configuration file `cmake/config_refactor.cmake`.

~~~bash
# 1. prepare compile system for `Hetu` (under root/hotspa folder)
cd hotspa
bash env/envs.sh
~~~

~~~bash
# 2. build and compile `Hetu` (under root folder)
mkdir -p build && cd build
cmake ..
make -j 32 _hetu_core
cd ..
source hetu_refactor.exp
~~~

~~~bash
# 3. some related python envs
pip3 install nvidia-ml-py3
pip3 install numpy
pip3 install tqdm
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
~~~

Now you can import `hetu` in python by `import hetu`.

### Dataset Preparation

In the `HotSPa` paper, we use two datasets: `commoncrawl` and `github`. You may prepare them by the following steps. (root_folder: `/hotspa`)

1. Download the `.parquet` format dataset from huggingface, such as `https://huggingface.co/datasets/bigcode/starcoderdata`

2. Convert the `.parquet` format dataset into `.json` format via `python3 data_utils/create_dataset.py`

3. Generate the dataset cache to speedup data loading via `python3 data_utils/llama_dataset.py`. 

for convenience, we provide a **one-click** script to generate a subset of commoncrawl dataset for testing

~~~bash
# prepare subset commoncrawl dataset for testing (under root/hotspa folder)
cd hotspa
bash data_utils/create_web_dataset.sh
~~~

### Training with Static (Hybrid) Parallelism Strategies

We have prepared the model definition and training codes/scripts in the `hotspa` directory. The training script (`hotspa/scripts/llama_static_parallel.sh`) will auto-gen & load a distributed configuration file (under `hotspa/ds_parallel_config`) that indicates the expected parallelism strategy, and launch the training code (`hotspa/llama_hot_switch.py`) to start the training.

*Our system further supports splitting tenors in any dimension other than the Megatron-style row/column parallelism. If you wish to explore more complex strategies other than 3d parallelism, you can define the distributed configuration files manually.*

After the distributed configurations files are ready, you can prepare a hostfile for MPI, and then launch training with a static parallelism strategy via

~~~bash
# hostfile example for 4xnodes, each node with 8xGPUs
22.22.22.22 slots=8
22.22.22.23 slots=8
22.22.22.24 slots=8
22.22.22.25 slots=8
~~~

~~~bash
cd hotspa
bash scripts/llama_static_parallel.sh [3b|7b|13b|32b] <context_len> <global batch size> <micro batch size> <hostfile for MPI> <steps> <epochs>
~~~

the static parallelism config is inside `llama_static_parallel.sh`, like:

~~~bash
BUCKET_SIZES=(4096 0) # means 0 <= seqlen <= 4096; others: (8192, 0), or (16384, 0) or (32768, 0)
ALL_PARALLEL_STRATEGY=("8,1,1") # means <DP, TP, PP>
# ALL_PARALLEL_STRATEGY=("4,2,1")
# ALL_PARALLEL_STRATEGY=("4,1,2")
# ALL_PARALLEL_STRATEGY=("2,4,1")
# ALL_PARALLEL_STRATEGY=("2,2,2")
# ALL_PARALLEL_STRATEGY=("1,4,2")
# ALL_PARALLEL_STRATEGY=("1,8,1")
~~~

### Training with Parallelism Hot Switching

Training with parallelism hot switching can be launched via training script (`hotspa/scripts/llama_hot_switch.sh`)

~~~bash
cd hotspa
bash scripts/llama_hot_switch.sh [3b|7b|13b|32b] <context_len> <global batch size> <hostfile for MPI> <steps> <epochs>
~~~

the multi parallelism configs for parallelism hot switching is inside `llama_hot_switch.sh`, like:

~~~bash
# for CommonCrawl Dataset
# BUCKET_SIZES=(32768 16384 4096 0)
BUCKET_SIZES=(16384 8192 4096 0) # means: 16384>=bucket0>=8192, 8192>=bucket1>=4096, 4096>=bucket2>=0
# BUCKET_SIZES=(4096 2048 1024 0)
# case1: 7B, 8GPUs
ALL_PARALLEL_STRATEGY=("1,4,2" "8,1,1" "2,2,2") # means: <DP1,TP4,PP2> for bucket0, <DP8,TP1,PP1> for bucket2, <DP2,TP2,PP2> for bucket1
# # case2: 7B, 16GPUs
# ALL_PARALLEL_STRATEGY=("2,4,2" "16,1,1" "4,4,1")
# # case3: 13B, 8GPUs
# ALL_PARALLEL_STRATEGY=("1,8,1" "4,2,1" "1,4,2")
# # case4: 13B, 16GPUs
# ALL_PARALLEL_STRATEGY=("2,8,1" "8,2,1" "4,4,1")
# # case5: 32B, 32GPUs
# ALL_PARALLEL_STRATEGY=("1,16,2" "4,2,4" "2,8,2")

# # for GitHub Dataset
# # case1: 7B, 8GPUs
# ALL_PARALLEL_STRATEGY=("1,4,2" "8,1,1" "4,2,1" "2,2,2")
# # case2: 7B, 16GPUs
# ALL_PARALLEL_STRATEGY=("2,4,2" "16,1,1" "8,2,1" "4,4,1")
# # case3: 13B, 8GPUs
# ALL_PARALLEL_STRATEGY=("1,8,1" "4,2,1" "2,2,2" "1,4,2")
# # case4: 13B, 16GPUs
# ALL_PARALLEL_STRATEGY=("2,8,1" "8,2,1" "4,2,2" "4,4,1")
# # case5: 32B, 32GPUs
# ALL_PARALLEL_STRATEGY=("1,16,2" "4,2,4" "4,4,2" "2,8,2")
~~~
