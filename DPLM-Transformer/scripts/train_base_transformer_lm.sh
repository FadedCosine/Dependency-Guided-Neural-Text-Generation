MAX_TOKEN=10240
BATCH_SIZE=42
lr=0.0005
TEXT=rocstories
BREAK_MODE=eos
TOKEN_PER_SAMPLE=512
RESULT=checkpoints/$TEXT/transformer_lm
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ../data/$TEXT/data-bin \
  --task language_modeling \
  --sample-break-mode $BREAK_MODE --max-tokens $MAX_TOKEN \
  --save-dir $RESULT \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr "$lr" --lr-scheduler inverse_sqrt \
  --keep-last-epochs 3 \
  --distributed-world-size 2 \
  --ddp-backend legacy_ddp \
  --max-epoch 50 