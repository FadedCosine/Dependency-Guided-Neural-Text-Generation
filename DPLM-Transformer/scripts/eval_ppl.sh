TEXT=news
BREAK_MODE=eos
model=dpgpt2_$BREAK_MODE
TOKEN_PER_SAMPLE=512
BATCH_SIZE=32
CUDA_VISIBLE_DEVICES=0 fairseq-eval-lm ../data/$TEXT/gpt2-data-bin \
    --path checkpoints/$TEXT/$model/checkpoint35.pt \
    --skip-invalid-size-inputs-valid-test \
    --sample-break-mode $BREAK_MODE \
    --batch-size "$BATCH_SIZE" 