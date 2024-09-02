# C1
# 16 GPUs
# CommonCrawl: 7B, 16GPUs, 4K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 4096 512 hostfile_16GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 7B, 16GPUs, 4K end ======================================================"

# CommonCrawl: 7B, 16GPUs, 8K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 8192 512 hostfile_16GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 7B, 16GPUs, 8K end ======================================================"

# CommonCrawl: 7B, 16GPUs, 16K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 16384 512 hostfile_16GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 7B, 16GPUs, 16K end ======================================================"

# CommonCrawl: 7B, 16GPUs, 32K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 32768 512 hostfile_16GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 7B, 16GPUs, 32K end ======================================================"

# C2
# 16 GPUs
# CommonCrawl: 7B, 16GPUs, 4K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 4096 512 hostfile_16GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 7B, 16GPUs, 4K end ======================================================"

# CommonCrawl: 7B, 16GPUs, 8K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 8192 512 hostfile_16GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 7B, 16GPUs, 8K end ======================================================"

# CommonCrawl: 7B, 16GPUs, 16K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 16384 512 hostfile_16GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 7B, 16GPUs, 16K end ======================================================"

# CommonCrawl: 7B, 16GPUs, 32K
echo "====================================================== CommonCrawl: 7B, 16GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 7b 32768 512 hostfile_16GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 7B, 16GPUs, 32K end ======================================================"
