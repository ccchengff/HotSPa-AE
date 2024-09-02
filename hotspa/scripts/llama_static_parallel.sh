MODEL_SIZE=${1:-'3b'}
SEQ_LEN=${2:-4096}
if [ "${MODEL_SIZE}" = "3b" ]; then
        NUM_LAYERS=16
        HIDDEN_SIZE=2048
	    FFN_HIDDEN_SIZE=5120
        NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "7b" ]; then
        NUM_LAYERS=32
        HIDDEN_SIZE=4096
	    FFN_HIDDEN_SIZE=11008
        NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "13b" ]; then
        NUM_LAYERS=40
        HIDDEN_SIZE=5120
	    FFN_HIDDEN_SIZE=13824
        NUM_HEADS=40
elif [ "${MODEL_SIZE}" = "32b" ]; then
        # actually 30b = 12*num_layers*hidden_size^2
        NUM_LAYERS=60
        HIDDEN_SIZE=6656
	    FFN_HIDDEN_SIZE=17920
        NUM_HEADS=64
else
        echo the model should be 7b/13b/30b for test.
        exit 0
fi

GLOBAL_BATCH_SIZE=${3:-512}
MICRO_BATCH_SIZE=${4:-1} # only for static strategy, is useless for parallelism hot switching
HOSTFILE=${5:-'hostfile'}
STEPS=${6:-100}
EPOCHS=${7:-1}

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))
echo MODEL_SIZE = LLaMA-$MODEL_SIZE, NNODES = $NNODES, NUM_GPUS_PER_NODE = $NUM_GPUS_PER_NODE, WORLD_SIZE = $WORLD_SIZE

# for Static Strategy
BUCKET_SIZES=(${SEQ_LEN} 0) # 0<=seqlen<=4096
ALL_PARALLEL_STRATEGY=("8,1,1")
# ALL_PARALLEL_STRATEGY=("4,2,1")
# ALL_PARALLEL_STRATEGY=("4,1,2")
# ALL_PARALLEL_STRATEGY=("2,4,1")
# ALL_PARALLEL_STRATEGY=("2,2,2")
# ALL_PARALLEL_STRATEGY=("1,4,2")
# ALL_PARALLEL_STRATEGY=("1,8,1")

if [ "${BUCKET_SIZES[0]}" -ne "$SEQ_LEN" ]; then
    echo "BUCKET_SIZES[0] ${BUCKET_SIZES[0]} is not equal to SEQ_LEN ${SEQ_LEN}!!!"
    exit 1
fi

DS_PARALLEL_CONFIGS=()
echo MAX_CONTEXT_LENGTH = $SEQ_LEN, BUCKET_SIZES = $BUCKET_SIZES, ALL_PARALLEL_STRATEGY = $ALL_PARALLEL_STRATEGY
echo "##############################" generate DS_PARALLEL_CONFIGS begin "##############################"
for PARALLEL_STRATEGY in ${ALL_PARALLEL_STRATEGY[*]}; do
    IFS=',' read -r DP TP PP <<< $PARALLEL_STRATEGY
    IDX=${#DS_PARALLEL_CONFIGS[*]}
    echo PARALLEL_STRATEGY[$IDX]: DP=$DP, TP=$TP, PP=$PP

    NUM_GPUS=$(( $DP * $TP * $PP ))
    if [ ${NUM_GPUS} -ne ${WORLD_SIZE} ]; then
        echo PARALLEL_STRATEGY[$IDX]: world size ${WORLD_SIZE} is not equal to dp ${DP} x tp ${TP} x pp ${PP}!
        exit 0
    fi
    DS_PARALLEL_CONFIG=ds_parallel_config/gpus${NUM_GPUS}/${MODEL_SIZE}/dp${DP}_tp${TP}_pp${PP}.json
    if [ ! -f ${DS_PARALLEL_CONFIG} ]; then
        python3 ds_parallel_config/generate_llama_3d_config.py --model_size ${MODEL_SIZE} --num_gpus ${NUM_GPUS} --dp ${DP} --tp ${TP} --pp ${PP} --zero
        echo generate ${DS_PARALLEL_CONFIG}...
    else
        echo use cached ${DS_PARALLEL_CONFIG}...
    fi
    DS_PARALLEL_CONFIGS[$IDX]=${DS_PARALLEL_CONFIG}
done
NUM_STRATEGY=${#DS_PARALLEL_CONFIGS[*]}
echo DS_PARALLEL_CONFIGS=${DS_PARALLEL_CONFIGS[*]}
echo "##############################" generate DS_PARALLEL_CONFIGS end "##############################"


ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=WARN
export HETU_MEMORY_PROFILE=WARN

export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

pkill -9 python3 # kill origin python3 processes

mpirun --allow-run-as-root -np ${NUM_GPUS} --hostfile ${HOSTFILE} \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
    -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
    -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
    -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_MEMORY_PROFILE \
    --output-filename logs/ds_parallel --merge-stderr-to-stdout \
    python3 llama_hot_switch.py \
    --num_strategy=$NUM_STRATEGY \
    --ds_parallel_config "${DS_PARALLEL_CONFIGS[*]}" \
    --bucket_sizes "${BUCKET_SIZES[*]}" \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --json_file $JSON_FILE \
    --json_key $JSON_KEY \
    --vocab_file $VOCAB_FILE \
    --merge_file $MERGE_FILE \
    --vocab_size 30592 \
    --hidden_size $HIDDEN_SIZE \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --num_attention_heads $NUM_HEADS \
    --seq_length $SEQ_LEN \
    --epochs $EPOCHS \
    --steps $STEPS \
    --lr 1e-5 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --bf16 \
    --use_flash_attn \