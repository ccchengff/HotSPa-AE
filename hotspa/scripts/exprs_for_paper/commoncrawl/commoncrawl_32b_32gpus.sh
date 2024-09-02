# C1
# 32 GPUs
# CommonCrawl: 32B, 32GPUs, 4K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 4096 512 hostfile_32GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 32B, 32GPUs, 4K end ======================================================"

# CommonCrawl: 32B, 32GPUs, 8K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 8192 512 hostfile_32GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 32B, 32GPUs, 8K end ======================================================"

# CommonCrawl: 32B, 32GPUs, 16K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 16384 512 hostfile_32GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 32B, 32GPUs, 16K end ======================================================"

# CommonCrawl: 32B, 32GPUs, 32K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 32768 512 hostfile_32GPUs 100 1 web/commoncrawl.json c1
echo "====================================================== CommonCrawl: 32B, 32GPUs, 32K end ======================================================"

# C2
# 32 GPUs
# CommonCrawl: 32B, 32GPUs, 4K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 4K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 4096 512 hostfile_32GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 32B, 32GPUs, 4K end ======================================================"

# CommonCrawl: 32B, 32GPUs, 8K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 8K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 8192 512 hostfile_32GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 32B, 32GPUs, 8K end ======================================================"

# CommonCrawl: 32B, 32GPUs, 16K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 16K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 16384 512 hostfile_32GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 32B, 32GPUs, 16K end ======================================================"

# CommonCrawl: 32B, 32GPUs, 32K
echo "====================================================== CommonCrawl: 32B, 32GPUs, 32K begin ======================================================"
bash scripts/exprs_for_paper/llama_hot_switch.sh 32b 32768 512 hostfile_32GPUs 100 1 web/commoncrawl.json c2
echo "====================================================== CommonCrawl: 32B, 32GPUs, 32K end ======================================================"
