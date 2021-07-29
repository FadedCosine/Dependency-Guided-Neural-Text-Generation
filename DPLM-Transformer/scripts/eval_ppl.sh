model=dpt_fg_0_rdrop
TEXT=rocstories
BREAK_MODE=eos
TOKEN_PER_SAMPLE=512
BATCH_SIZE=128
CUDA_VISIBLE_DEVICES=7 fairseq-eval-lm ../data/$TEXT/data-bin \
    --path checkpoints/$TEXT/$model/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --sample-break-mode $BREAK_MODE \
    --batch-size "$BATCH_SIZE" 