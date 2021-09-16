pointer_layer=-2
MAX_TOKEN=5120
lr=0.0005
TEXT=news
BREAK_MODE=eos
TOKEN_PER_SAMPLE=512
BATCH_SIZE=16
RESULT=checkpoints/$TEXT/dp_transformer_fg0
DEP_MODEL_PATH=checkpoints/$TEXT/dependency_lm
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ../data/$TEXT/data-bin \
    --sample-break-mode $BREAK_MODE \
    --max-tokens $MAX_TOKEN \
    --save-dir $RESULT \
    --alignment-layer "$pointer_layer" \
    --dependency-model-path $DEP_MODEL_PATH \
    --dependency-model-filename checkpoint_best.pt \
    --data-path-for-load-model ../data/$TEXT/data-bin \
    --task language_modeling \
    --arch dependency_pointer_only_transformer_lm_gpt \
    --share-decoder-input-output-embed \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr "$lr" --lr-scheduler inverse_sqrt \
    --keep-last-epochs 3 \
    --distributed-world-size 4 \
    --ddp-backend legacy_ddp \
    --max-epoch 30 \
    --force-generation 0