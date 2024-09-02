# C1
# 32 GPUs
# GitHub: 32B, 32GPUs, 4K
echo "====================================================== GitHub: 32B, 32GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 4096 512 hostfile_32GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 32B, 32GPUs, 4K end ======================================================"

# GitHub: 32B, 32GPUs, 8K
echo "====================================================== GitHub: 32B, 32GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 8192 512 hostfile_32GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 32B, 32GPUs, 8K end ======================================================"

# GitHub: 32B, 32GPUs, 16K
echo "====================================================== GitHub: 32B, 32GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 16384 512 hostfile_32GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 32B, 32GPUs, 16K end ======================================================"

# GitHub: 32B, 32GPUs, 32K
echo "====================================================== GitHub: 32B, 32GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 32768 512 hostfile_32GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 32B, 32GPUs, 32K end ======================================================"

# C2
# 32 GPUs
# GitHub: 32B, 32GPUs, 4K
echo "====================================================== GitHub: 32B, 32GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 4096 512 hostfile_32GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 32B, 32GPUs, 4K end ======================================================"

# GitHub: 32B, 32GPUs, 8K
echo "====================================================== GitHub: 32B, 32GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 8192 512 hostfile_32GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 32B, 32GPUs, 8K end ======================================================"

# GitHub: 32B, 32GPUs, 16K
echo "====================================================== GitHub: 32B, 32GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 16384 512 hostfile_32GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 32B, 32GPUs, 16K end ======================================================"

# GitHub: 32B, 32GPUs, 32K
echo "====================================================== GitHub: 32B, 32GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 32768 512 hostfile_32GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 32B, 32GPUs, 32K end ======================================================"
