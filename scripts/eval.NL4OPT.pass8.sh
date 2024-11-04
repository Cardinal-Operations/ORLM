MODEL_PATH=$1
NUM_GPUS=$2

TEST_DATASET_NAME="CardinalOperations/NL4OPT"
TEST_DATASET_SPLIT="test"

Q2MC_OUTPUT_DIR="$MODEL_PATH/eval.NL4OPT.pass8"

export HF_ENDPOINT=https://hf-mirror.com

python -m eval.generate \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $TEST_DATASET_NAME \
    --dataset_split $TEST_DATASET_SPLIT \
    --tensor_parallel_size $NUM_GPUS \
    --save_dir $Q2MC_OUTPUT_DIR \
    --topk 8 \
    --decoding sampling \
    --verbose

python -m eval.execute \
    --input_file $Q2MC_OUTPUT_DIR/generated.jsonl \
    --output_file $Q2MC_OUTPUT_DIR/executed.jsonl \
    --question_field en_question \
    --answer_field en_answer \
    --timeout 600 \
    --max_workers 16 \
    --verbose