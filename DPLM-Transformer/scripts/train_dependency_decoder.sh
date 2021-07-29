MAX_TOKEN=8192
TEXT=news
lr=0.0005
BREAK_MODE=none
TOKEN_PER_SAMPLE=70
BATCH_SIZE=200
RESULT=/home/yangzhixian/DependencyGuided/DPLM-Transformer/checkpoints/$TEXT/dependency_lm_CE_none_debug
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 \
    /home/yangzhixian/fairseq/fairseq_cli/train.py \
    --task dependency_modeling /home/yangzhixian/DependencyGuided/data/$TEXT/data-bin \
    --sample-break-mode $BREAK_MODE \
    --batch-size "$BATCH_SIZE" \
    --tokens-per-sample $TOKEN_PER_SAMPLE \
    --save-dir $RESULT \
    --dependency /home/yangzhixian/DependencyGuided/data/$TEXT/dependency --dependency-suffix .head \
    --arch transformer_lm --share-decoder-input-output-embed \
    --criterion dependency_cross_entropy \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 --lr "$lr" --lr-scheduler inverse_sqrt \
    --keep-last-epochs 5 \
    --num-workers 1 \
    --max-epoch 100  
    
    # --max-tokens $MAX_TOKEN \
    # --distributed-world-size 4 \
    # --ddp-backend legacy_ddp \
    
    