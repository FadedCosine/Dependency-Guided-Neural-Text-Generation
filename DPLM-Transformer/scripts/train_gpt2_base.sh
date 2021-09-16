TEXT=ptb
MAX_TOKEN=2048
TOKEN_PER_SAMPLE=512
BATCH_SIZE=4
BREAK_MODE=eos
WARMUP_UPDATES=0
LR=5e-5
RESULT=checkpoints/$TEXT/gpt2_$BREAK_MODE
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ../data/$TEXT/gpt2-data-bin \
    --task language_modeling \
    --save-dir $RESULT \
    --arch hf_gpt2 --pretrained-model ~/pretrained_model/gpt2-base \
    --sample-break-mode $BREAK_MODE \
    --max-tokens $MAX_TOKEN \
    --save-dir $RESULT \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-8  \
    --weight-decay 0.00 --clip-norm 0.0 \
    --lr $LR --warmup-updates $WARMUP_UPDATES  \
    --distributed-world-size 2 \
    --ddp-backend legacy_ddp \
    --keep-last-epochs 2 \
    --update-freq 4 \
    --max-epoch 80