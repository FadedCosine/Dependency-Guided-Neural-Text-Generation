MAX_TOKEN=8192
TEXT=news
lr=0.0005
BREAK_MODE=eos
TOKEN_PER_SAMPLE=70
BATCH_SIZE=200
RESULT=checkpoints/$TEXT/dependency_lm
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    ~/fairseq/fairseq_cli/train.py \
    --task dependency_modeling ../data/$TEXT/data-bin \
    --sample-break-mode $BREAK_MODE \
    --max-tokens $MAX_TOKEN \
    --save-dir $RESULT \
    --dependency ../data/$TEXT/dependency --dependency-suffix .head \
    --arch transformer_lm --share-decoder-input-output-embed \
    --criterion dependency_cross_entropy \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 --lr "$lr" --lr-scheduler inverse_sqrt \
    --keep-last-epochs 5 \
    --distributed-world-size 4 \
    --ddp-backend legacy_ddp \
    --num-workers 1 \
    --max-epoch 30  
