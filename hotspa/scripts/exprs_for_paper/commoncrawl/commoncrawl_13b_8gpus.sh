# C1
# 8 GPUs
# CommonCrawl: 13B, 8GPUs, 4K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 4096 512 hostfile_8GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 13B, 8GPUs, 4K end ======================================================"

# CommonCrawl: 13B, 8GPUs, 8K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 8192 512 hostfile_8GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 13B, 8GPUs, 8K end ======================================================"

# CommonCrawl: 13B, 8GPUs, 16K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 16384 512 hostfile_8GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 13B, 8GPUs, 16K end ======================================================"

# CommonCrawl: 13B, 8GPUs, 32K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 32768 512 hostfile_8GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 13B, 8GPUs, 32K end ======================================================"

# C2
# 8 GPUs
# CommonCrawl: 13B, 8GPUs, 4K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 4096 512 hostfile_8GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 13B, 8GPUs, 4K end ======================================================"

# CommonCrawl: 13B, 8GPUs, 8K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 8192 512 hostfile_8GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 13B, 8GPUs, 8K end ======================================================"

# CommonCrawl: 13B, 8GPUs, 16K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 16384 512 hostfile_8GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 13B, 8GPUs, 16K end ======================================================"

# CommonCrawl: 13B, 8GPUs, 32K
echo "====================================================== CommonCrawl: 13B, 8GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 13b 32768 512 hostfile_8GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 13B, 8GPUs, 32K end ======================================================"
