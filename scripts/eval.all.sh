MODEL_PATH=$1
NUM_GPUS=$2

sh scripts/eval.NL4OPT.pass1.sh $MODEL_PATH $NUM_GPUS
sh scripts/eval.MAMO_EasyLP.pass1.sh $MODEL_PATH $NUM_GPUS
sh scripts/eval.MAMO_ComplexLP.pass1.sh $MODEL_PATH $NUM_GPUS
sh scripts/eval.IndustryOR.pass1.sh $MODEL_PATH $NUM_GPUS

sh scripts/eval.NL4OPT.pass8.sh $MODEL_PATH $NUM_GPUS
sh scripts/eval.MAMO_EasyLP.pass8.sh $MODEL_PATH $NUM_GPUS
sh scripts/eval.MAMO_ComplexLP.pass8.sh $MODEL_PATH $NUM_GPUS
sh scripts/eval.IndustryOR.pass8.sh $MODEL_PATH $NUM_GPUS

python scripts/read_eval_results.py $MODEL_PATH