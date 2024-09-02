# C1
# 8 GPUs
# GitHub: 13B, 8GPUs, 4K
echo "====================================================== GitHub: 13B, 8GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 4096 512 hostfile_8GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 13B, 8GPUs, 4K end ======================================================"

# GitHub: 13B, 8GPUs, 8K
echo "====================================================== GitHub: 13B, 8GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 8192 512 hostfile_8GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 13B, 8GPUs, 8K end ======================================================"

# GitHub: 13B, 8GPUs, 16K
echo "====================================================== GitHub: 13B, 8GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 16384 512 hostfile_8GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 13B, 8GPUs, 16K end ======================================================"

# GitHub: 13B, 8GPUs, 32K
echo "====================================================== GitHub: 13B, 8GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 32768 512 hostfile_8GPUs 100 1 code/github.json c1
echo "====================================================== GitHub: 13B, 8GPUs, 32K end ======================================================"

# C2
# 8 GPUs
# GitHub: 13B, 8GPUs, 4K
echo "====================================================== GitHub: 13B, 8GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 4096 512 hostfile_8GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 13B, 8GPUs, 4K end ======================================================"

# GitHub: 13B, 8GPUs, 8K
echo "====================================================== GitHub: 13B, 8GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 8192 512 hostfile_8GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 13B, 8GPUs, 8K end ======================================================"

# GitHub: 13B, 8GPUs, 16K
echo "====================================================== GitHub: 13B, 8GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 16384 512 hostfile_8GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 13B, 8GPUs, 16K end ======================================================"

# GitHub: 13B, 8GPUs, 32K
echo "====================================================== GitHub: 13B, 8GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 32768 512 hostfile_8GPUs 100 1 code/github.json c2
echo "====================================================== GitHub: 13B, 8GPUs, 32K end ======================================================"
